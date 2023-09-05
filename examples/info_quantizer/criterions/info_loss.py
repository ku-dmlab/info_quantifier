# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from xmlrpc.client import Boolean
import tensorboardX
import math

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("info_loss")
class InfoLossCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        report_edit_distance=False,
        early_update=False,
        agent_arch=None,
        agent_loss=None,
        use_length_loss=True,
        length_loss_coef=0.3,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.report_edit_distance = report_edit_distance
        self.early_update = early_update
        self.agent_arch = agent_arch
        self.agent_loss = agent_loss
        self.use_length_loss = use_length_loss
        self.length_loss_coef = length_loss_coef
        self.pad_idx = task.src_dict.pad_index
        self.diff_loss_target = 0.3

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--report-edit-distance', action='store_true',
                            help='report Hamming and Levenshtein distances')
        parser.add_argument('--early-update', action='store_true',
                            help='update the loss based the first mistake and ignores the rest of the sequence')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        parser.add_argument('--agent-arch', default = None, type=str,
                            help='Agent Arch ["softplus"]')
        parser.add_argument('--agent-loss', default = None ,type=str,
                            help='Agent Loss ["Relu,Leaky Relu,Elu"]')
        parser.add_argument('--use-length-loss', default = True ,type=Boolean,
                            help='Use length loss or not')
        parser.add_argument('--length-loss-coef', default = 0.3 ,type=float,
                            help='Length loss hyperparam')
        
        # fmt: on

    def forward(self, model, sample, early_update=False, reduce=True):
        """Compute the loss for the given sample.

        We need to passs early_update parameter to make sure we won't apply it during validation

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
            
        net_output = model(sample["net_input"])
        bsz = sample['target'].size(-1)
        ## MAKE CONDITINOS 
        ## 1: Simgoid => SUM (Heuristic Idea is contiained)
        # NOT WORKING BECAUSE OUTPUT GOES TO -inf 
        ## 1-1: Softplus +1 => sum
        out_src_info = net_output[0]["src_info"].view(bsz, -1)
        out_trg_info = net_output[0]["trg_info"].view(bsz, -1)

        src_mask = (sample['net_input']['src_tokens'] == self.pad_idx)
        trg_mask = (sample['net_input']['trg_tokens'] == self.pad_idx)

        src_info, trg_info = self.sum_info(
            src_info=out_src_info, 
            trg_info=out_trg_info,
            src_mask=src_mask,
            trg_mask=trg_mask,
        )
        valid_info = src_info - trg_info
        
        ## 2: SUM => Sigmoid (Heuristic Idea is not contained)
        
        ## 3: Check info conditions
        target = sample['target'].float()
        target = sample['target'] - 4. # READ = 0 , WRITE = 1 PAD = -3
        assert (target <= 1).all()
        # valid_info = torch.where(target == -3. , 0.0, valid_info)
        assert (target.ne(0.) | target.ne(1.)).all()
        valid_info = torch.where(target == 1. , -valid_info, valid_info)
        valid_info_clone = valid_info.clone()
        
        ## 4: CHoose Loss Function: Default is RELU / Test other functions later 
        loss = self.loss_info(valid_info=valid_info)
        
        ## 5: Additional Length Loss
        if self.use_length_loss:
            sum_info = src_info + trg_info
            src_lens = sample['net_input']['src_lengths']
            trg_lens = sample['net_input']['trg_lengths']
            total_lens = src_lens + trg_lens

            # ------- DH -----------
            # diff_info = (src_info - trg_info) ** 2
            # difference_loss = self.diff_loss(diff_info=diff_info)
            # loss = loss + self.length_loss_coef * difference_loss
            # ----------------------
            # ---- Original MK ----------
            target_mask = (target == -3.)
            lengths_loss = self.length_loss(sum_info, total_lens)

            loss = loss + self.length_loss_coef * lengths_loss
            # --------------------------

        # sample_size = (
        #     bsz if self.sentence_avg else sample["ntokens"]
        # )
        sample_size = bsz
        logging_output = {
            "loss": loss.data * bsz,
            "nll_loss": 0,
            "ntokens": sample["ntokens"],
            "nsentences": bsz,
            "sample_size": bsz,
            # "sample_size": sample_size,
            # "accuracy": (valid_info_clone < 0).sum() / (valid_info_clone.size(0)),
            "total": bsz,
            "n_correct": (valid_info_clone < 0).sum()
        }
        return loss, sample_size, logging_output

    def sum_info(self, src_info, trg_info, src_mask=None, trg_mask=None):           
        if self.agent_arch == "softplus":
            src_info_bias = (~src_mask).bool()
            # trg_info_bias = (~src_mask).bool()

            src_info = torch.nn.functional.softplus(src_info) + src_info_bias
            trg_info = torch.nn.functional.softplus(trg_info)
            
            src_info.masked_fill_(src_mask, 0)
            trg_info.masked_fill_(trg_mask, 0)
            
            src_info = torch.sum(src_info,dim=-1).squeeze()
            trg_info = torch.sum(trg_info,dim=-1).squeeze()
            # violation = src_info - trg_info
        elif self.agent_arch == "softplus2":
            # src_info_bias = ~(src_info == 0)
            # trg_info_bias = ~(trg_info == 0)

            # src_info.masked_fill_(src_mask, 0)
            # trg_info.masked_fill_(trg_mask, 0)

            src_info = torch.nn.functional.softplus(src_info) #+ src_info_bias
            trg_info = torch.nn.functional.softplus(trg_info) #+ trg_info_bias
            
            src_info.masked_fill_(src_mask, 0)
            trg_info.masked_fill_(trg_mask, 0)
            
            src_info = torch.sum(src_info, dim=-1).squeeze()
            trg_info = torch.sum(trg_info, dim=-1).squeeze()
            # violation = src_info - trg_info
        else:
            raise NotImplementedError
        return src_info, trg_info
    
    def loss_info(self, valid_info):
        # TODO: Difference Using Mean and Sum? (Scale diff?)
        if self.agent_loss == "relu":
            return torch.relu(valid_info).mean() #.sum()
        elif self.agent_loss =="relu2":
            return (torch.relu(valid_info).mean())**2
        elif self.agent_loss == "leakyrelu":
            return torch.nn.functional.leaky_relu(valid_info).mean() # .mean()
        elif self.agent_loss == "elu":
            return torch.nn.functional.elu(valid_info).mean()
        else:
            raise NotImplementedError
    
    def length_loss(self, sum_info, total_lens):
        # total_lens.masked_fill_(target_mask, 0)
        total_lens = total_lens.float()
        len_loss = torch.mean((sum_info - total_lens)**2)
        return len_loss
        # total_info_target = torch.arange(target.size(1)).repeat(target.size(0)).view(target.size()).to(target.device) + 2 # Add 2 because Start with Read 
        # length_info_target = torch.where(target==-3,0,total_info_target).float()
        # len_loss = torch.mean((sum_info - length_info_target)**2)
        # return len_loss
        
    def diff_loss(self, diff_info):
        difference_loss = torch.relu(self.diff_loss_target - diff_info).mean()
        return difference_loss
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
