import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor
from examples.info_quantizer.models import waitk_transformer


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        self.has_incremental: bool = False
        if all(
                hasattr(m, "decoder") and isinstance(m.decoder, FairseqIncrementalDecoder)
                for m in models
        ):
            self.has_incremental = True

    def forward(self, net_input: Dict[str, Tensor]):
        return [model(**net_input) for model in self.models]

    def has_encoder(self):
        return hasattr(self.single_model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min([m.max_decoder_positions() for m in self.models])

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_agent_decoder(
            self,
            tokens,
            prev_output_tokens,
            incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
            temperature: float = 1.0,
            has_incremental=True,
    ):
        log_probs = []
        self.has_incremental = has_incremental
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.forward(
                    tokens,
                    prev_output_tokens,
                    incremental_state=incremental_states[i],
                )
                decoder_out_tuple = (
                    decoder_out[0][:, -1:, :].div_(temperature),
                    None if len(decoder_out) <= 1 else decoder_out[1],
                )
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)
                decoder_out_tuple = (
                    decoder_out[0][:, :, :].div_(temperature),
                    None if len(decoder_out) <= 1 else decoder_out[1],
                )

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None and self.has_incremental_states():
                    attn = attn[:, -1, :]


            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )

            if self.has_incremental_states():
                probs = probs[:, -1, :]

            if self.models_size == 1:
                return probs, attn

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def forward_decoder(
            self,
            tokens,
            encoder_outs: List[Dict[str, List[Tensor]]],
            incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
            temperature: float = 1.0,
            has_incremental=True,
    ):
        log_probs = []
        self.has_incremental = has_incremental
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                )
                decoder_out_tuple = (
                    decoder_out[0][:, -1:, :].div_(temperature),
                    None if len(decoder_out) <= 1 else decoder_out[1],
                )
            else:
                decoder_out = model.decoder.forward_train(tokens, encoder_out=encoder_out) \
                    if isinstance(model, waitk_transformer.WaitkTransformerModel) \
                    else model.decoder.forward(tokens, encoder_out=encoder_out)
                decoder_out_tuple = (
                    decoder_out[0][:, :, :].div_(temperature),
                    None if len(decoder_out) <= 1 else decoder_out[1],
                )

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None and self.has_incremental_states():
                    attn = attn[:, -1, :]


            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )

            if self.has_incremental_states():
                probs = probs[:, -1, :]

            if self.models_size == 1:
                return probs, attn

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def forward_decoder_features_only(
            self,
            tokens,
            encoder_outs: List[Dict[str, List[Tensor]]],
            incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            has_incremental=True,
    ):
        self.has_incremental = has_incremental
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                    features_only=True
                )
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out, features_only=True)

        return decoder_out[0]

    @torch.jit.export
    def reorder_encoder_out(self, encoder_outs: Optional[List[Dict[str, List[Tensor]]]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[Dict[str, List[Tensor]]] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    @torch.jit.export
    def reorder_tokens(self, encoder_outs: Optional[Tensor], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[Dict[str, List[Tensor]]] = []
        assert encoder_outs is not None
        new_outs.append(
            encoder_outs.index_select(0, new_order)
        )
        return new_outs

    def reorder_list(self, hypos, new_order):
        new_outs = [hypos[index] for index in new_order]
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(
            self,
            incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
            new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order
            )