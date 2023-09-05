import numbers
import sacrebleu
import torch
import torch.nn.functional as F
from torch import arange, cat
from fairseq import tasks
import numpy as np

def latency_metric(func):
    def prepare_latency_metric(
        delays,
        src_lens,
        ref_lens=None,
        target_padding_mask=None,
    ):
        """
        delays: bsz, tgt_len
        src_lens: bsz
        target_padding_mask: bsz, tgt_len
        """
        if isinstance(delays, list):
            delays = torch.FloatTensor(delays).unsqueeze(0)

        if len(delays.size()) == 1:
            delays = delays.view(1, -1)

        if isinstance(src_lens, list):
            src_lens = torch.FloatTensor(src_lens)
        if isinstance(src_lens, numbers.Number):
            src_lens = torch.FloatTensor([src_lens])
        if len(src_lens.size()) == 1:
            src_lens = src_lens.view(-1, 1)
        src_lens = src_lens.type_as(delays)

        if ref_lens is not None:
            if isinstance(ref_lens, list):
                ref_lens = torch.FloatTensor(ref_lens)
            if isinstance(ref_lens, numbers.Number):
                ref_lens = torch.FloatTensor([ref_lens])
            if len(ref_lens.size()) == 1:
                ref_lens = ref_lens.view(-1, 1)
            ref_lens = ref_lens.type_as(delays)

        if target_padding_mask is not None:
            tgt_lens = delays.size(-1) - target_padding_mask.sum(dim=1)
            delays = delays.masked_fill(target_padding_mask, 0)
        else:
            tgt_lens = torch.ones_like(src_lens) * delays.size(1)

        tgt_lens = tgt_lens.view(-1, 1)

        return delays, src_lens, tgt_lens, ref_lens, target_padding_mask

    def latency_wrapper(
        delays, src_lens, ref_lens=None, target_padding_mask=None
    ):
        delays, src_lens, tgt_lens, ref_lens, target_padding_mask = prepare_latency_metric(
            delays, src_lens, ref_lens, target_padding_mask)
        return func(delays, src_lens, tgt_lens, ref_lens, target_padding_mask)

    return latency_wrapper


def AverageProportion(
    delays, src_lens, tgt_lens, ref_lens=None, target_padding_mask=None):
    """
    Function to calculate Average Proportion from
    Can neural machine translation do simultaneous translation?
    (https://arxiv.org/abs/1606.02012)
    Delays are monotonic steps, range from 1 to src_len.
    Give src x tgt y, AP is calculated as:
    AP = 1 / (|x||y]) sum_i^|Y| delays_i
    """
    if target_padding_mask is not None:
        AP = torch.sum(delays.masked_fill(target_padding_mask, 0), dim=1, keepdim=True)
    else:
        AP = torch.sum(delays, dim=1, keepdim=True)

    AP = AP / (src_lens * tgt_lens)
    return AP.squeeze(1)


def AverageLagging(delays, src_lens, tgt_lens, ref_lens=None, target_padding_mask=None):
    """
    Function to calculate Average Lagging from
    STACL: Simultaneous Translation with Implicit Anticipation
    and Controllable Latency using Prefix-to-Prefix Framework
    (https://arxiv.org/abs/1810.08398)
    Delays are monotonic steps, range from 1 to src_len.
    Give src x tgt y, AP is calculated as:
    AL = 1 / tau sum_i^tau delays_i - (i - 1) / gamma
    Where
    gamma = |y| / |x|
    tau = argmin_i(delays_i = |x|)

    When reference was given, |y| would be the reference length
    """
    bsz, max_tgt_len = delays.size()
    if ref_lens is not None:
        max_tgt_len = ref_lens.max().long()
        tgt_lens = ref_lens

    # tau = argmin_i(delays_i = |x|)
    # Only consider the delays that has already larger than src_lens
    lagging_padding_mask = delays >= src_lens
    # Padding one token at beginning to consider at least one delays that
    # larget than src_lens
    lagging_padding_mask = torch.nn.functional.pad(
        lagging_padding_mask, (1, 0))[:, :-1]

    if target_padding_mask is not None:
        lagging_padding_mask = lagging_padding_mask.masked_fill(
            target_padding_mask, True)

    # oracle delays are the delay for the oracle system which goes diagonally
    oracle_delays = (
        torch.arange(max_tgt_len)
        .unsqueeze(0)
        .type_as(delays)
        .expand([bsz, max_tgt_len])
    ) * src_lens / tgt_lens

    if delays.size(1) < max_tgt_len:
        oracle_delays = oracle_delays[:, :delays.size(1)]

    if delays.size(1) > max_tgt_len:
        oracle_delays = torch.cat(
            [
                oracle_delays,
                oracle_delays[:,-1]
                * oracle_delays.new_ones(
                    [delays.size(0), delays.size(1) - max_tgt_len]
                )
            ],
            dim=1
        )

    lagging = delays - oracle_delays
    lagging = lagging.masked_fill(lagging_padding_mask, 0)

    # tau is the cut-off step
    tau = (1 - lagging_padding_mask.type_as(lagging)).sum(dim=1)
    AL = lagging.sum(dim=1) / tau

    return AL


def DifferentiableAverageLagging(
    delays, src_lens, tgt_lens, ref_len=None, target_padding_mask=None
):
    """
    Function to calculate Differentiable Average Lagging from
    Monotonic Infinite Lookback Attention for Simultaneous Machine Translation
    (https://arxiv.org/abs/1906.05218)
    """

    _, max_tgt_len = delays.size()

    gamma = tgt_lens / src_lens
    new_delays = torch.zeros_like(delays)

    for i in range(max_tgt_len):
        if i == 0:
            new_delays[:, i] = delays[:, i]
        else:
            new_delays[:, i] = torch.cat(
                [
                    new_delays[:, i - 1].unsqueeze(1) + 1 / gamma,
                    delays[:, i].unsqueeze(1)
                ],
                dim=1
            ).max(dim=1).values

    DAL = (
        new_delays -
        torch.arange(max_tgt_len).unsqueeze(0).type_as(
            delays).expand_as(delays) / gamma
    )
    if target_padding_mask is not None:
        DAL = DAL.masked_fill(target_padding_mask, 0)

    DAL = DAL.sum(dim=1, keepdim=True) / tgt_lens

    return DAL.squeeze(1)


def length_adaptive_average_lagging(delays, src_lens, tgt_lens, ref_lens=None, target_padding_mask=None):
    """
    Function to calculate Length Adaptive Average Lagging from
    "Over-Generation Cannot Be Rewarded:
    Length-Adaptive Average Lagging for Simultaneous Speech Translation"
    AutoSimTrans Workshop @ NAACL2022
    It modifies the original AL implementation of SimulEval tool
    by taking the maximum between the prediction and the reference length
    in computing the oracle delays to avoid favorable computation for
    under-generative or over-generative models.
    """
    bsz, max_tgt_len = delays.size() # hypo len

    # Take the maximum between the reference and prediction length to obtain LAAL
    if ref_lens is not None and ref_lens > max_tgt_len:
        max_tgt_len = ref_lens.max().long()
        tgt_lens = ref_lens

    # Only consider the delays that has already larger than src_lens
    lagging_padding_mask = delays >= src_lens
    # Shift left padding to consider one delay that is larger than src_lens
    lagging_padding_mask = F.pad(lagging_padding_mask, (1, 0))[:, :-1]

    if target_padding_mask is not None:
        lagging_padding_mask = lagging_padding_mask.masked_fill(
            target_padding_mask, True)

    # Oracle delays are the delay for the oracle system which goes diagonally
    oracle_delays = (
                        arange(max_tgt_len)
                        .unsqueeze(0)
                        .type_as(delays)
                        .expand([bsz, max_tgt_len])
                    ) * src_lens / tgt_lens

    if delays.size(1) < max_tgt_len:
        oracle_delays = oracle_delays[:, :delays.size(1)]

    if delays.size(1) > max_tgt_len:
        oracle_delays = cat(
            [
                oracle_delays,
                oracle_delays[:, -1]
                * oracle_delays.new_ones(
                    [delays.size(0), delays.size(1) - max_tgt_len]
                )
            ],
            dim=1
        )

    lagging = delays - oracle_delays
    lagging = lagging.masked_fill(lagging_padding_mask, 0)

    # tau is the cut-off step
    tau = (1 - lagging_padding_mask.type_as(lagging)).sum(dim=1)
    al = lagging.sum(dim=1) / tau
    
    # if al.item() < 0 and max_tgt_len < tgt_lens.item():
    #     print()

    return al
