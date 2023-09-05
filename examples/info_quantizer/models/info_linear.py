# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqLanguageModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.lstm import Embedding
from fairseq.modules import AdaptiveSoftmax, FairseqDropout
from torch import Tensor


DEFAULT_MAX_TARGET_POSITIONS = 1e5

from fairseq.dataclass import FairseqDataclass
from dataclasses import dataclass, field
from omegaconf import II

# @dataclass
# class InfoLinearConfig(FairseqDataclass):
#     # defaults come from the original Transformer-XL code
#     dropout: float = field(default=0.0)
#     decoder_embed_dim: int = field(default=512)
#     decoder_embed_path: str = field(default=None)
#     decoder_hidden_size: int = field(default=512)
#     decoder_layers: int = field(default=6)
#     decoder_out_embed_dim: int = field(default=512)
#     decoder_attention: str = field(default=False)
#     adaptive_softmax_cutoff: List[int] = field(default_factory=lambda: [20000, 40000, 200000])
#     residuals: bool = field(default=False)
#     decoder_dropout_int: float = field(default=0.4)
#     decoder_dropout_out: float = field(default=0.4)
#     share_decoder_input_output_embed: bool = field(default=False)
#     # cutoffs: List[int] = field(default_factory=lambda: [20000, 40000, 200000])
#     # max_target_positions: int = II("task.max_target_positions")

@register_model("info_linear")
# class InfoLinearModel(FairseqLanguageModel, dataclass=InfoLinearConfig):
class InfoLinearModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
                            help='decoder attention')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--residuals', default=False,
                            action='store_true',
                            help='applying residuals between LSTM layers')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')
        
        parser.add_argument('--layer-size', type=str, default='normal') # big
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if getattr(args, "max_target_positions", None) is not None:
            max_target_positions = args.max_target_positions
        else:
            max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        pretrained_decoder_embed = None
        if args.decoder_embed_path:
            pretrained_decoder_embed = load_pretrained_embedding_from_file(
                args.decoder_embed_path, task.target_dictionary, args.decoder_embed_dim
            )

        if args.share_decoder_input_output_embed:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError(
                    "--share-decoder-input-output-embeddings requires a joint dictionary"
                )

            if args.decoder_embed_dim != args.decoder_out_embed_dim:
                raise ValueError(
                    "--share-decoder-input-output-embeddings requires "
                    "--decoder-embed-dim to match --decoder-out-embed-dim"
                )
        layer_size = getattr(args, 'layer_size', 'normal')

        decoder = Decoder(
            src_dictionary=task.source_dictionary,
            trg_dictionary=task.target_dictionary,
            agt_dictionary=task.agent_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=False,  # decoder-only language model doesn't support attention
            encoder_output_units=0,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == "adaptive_loss"
                else None
            ),
            max_target_positions=max_target_positions,
            residuals=args.residuals,
            use_feature=task.use_transformer_feature,
            layer_size=layer_size,
        )

        return cls(decoder)

    def forward(self, sample, **kwargs):
        return self.decoder(sample, **kwargs)


class Decoder(FairseqIncrementalDecoder):
    def __init__(
            self,
            src_dictionary,
            trg_dictionary,
            agt_dictionary,
            embed_dim=512,
            hidden_size=512,
            out_embed_dim=512,
            num_layers=1,
            dropout_in=0.1,
            dropout_out=0.1,
            attention=True,
            encoder_output_units=512,
            pretrained_embed=None,
            share_input_output_embed=False,
            adaptive_softmax_cutoff=None,
            max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
            residuals=False,
            use_feature=False,
            layer_size='normal'
    ):
        super().__init__(agt_dictionary)
        self.dropout_in_module = FairseqDropout(
            dropout_in, module_name=self.__class__.__name__
        )
        self.dropout_out_module = FairseqDropout(
            dropout_out, module_name=self.__class__.__name__
        )
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.max_target_positions = max_target_positions
        self.residuals = residuals
        self.num_layers = num_layers

        self.adaptive_softmax = None
        src_num_embeddings = len(src_dictionary)
        trg_num_embeddings = len(trg_dictionary)
        # agt_num_embeddings = len(agt_dictionary)

        src_padding_idx = src_dictionary.pad()
        trg_padding_idx = trg_dictionary.pad()
        # agt_padding_idx = agt_dictionary.pad()

        self.use_feature = use_feature
        if not self.use_feature:
            self.src_embed_tokens = Embedding(src_num_embeddings, embed_dim, src_padding_idx)
            self.trg_embed_tokens = Embedding(trg_num_embeddings, embed_dim, trg_padding_idx)

        # self.agt_embed_tokens = Embedding(agt_num_embeddings, embed_dim, agt_padding_idx)

        if layer_size == 'normal':
            self.src_layers = nn.Sequential(
                    Linear(embed_dim, embed_dim , bias=False),
                    nn.Tanh(),
                    Linear(embed_dim, embed_dim, bias=False),
                    nn.Tanh(),
                    Linear(embed_dim, embed_dim, bias=False),
                    nn.Tanh(),
                    Linear(embed_dim, 1, bias=False),
                    # nn.Sigmoid(),   
            )
            self.trg_layers = nn.Sequential(
                    Linear(embed_dim, embed_dim, bias=False),
                    nn.Tanh(),
                    Linear(embed_dim, embed_dim, bias=False),
                    nn.Tanh(),
                    Linear(embed_dim, embed_dim, bias=False),
                    nn.Tanh(),
                    Linear(embed_dim, 1, bias=False),
                    # nn.Sigmoid(),   
            )
        elif layer_size == 'big':
            self.src_layers = nn.Sequential(
                    Linear(embed_dim, embed_dim * 2 , bias=False),
                    nn.Tanh(),
                    Linear(embed_dim * 2, embed_dim * 2, bias=False),
                    nn.Tanh(),
                    Linear(embed_dim * 2, 1, bias=False),
                    # nn.Sigmoid(),   
            )
            self.trg_layers = nn.Sequential(
                    Linear(embed_dim, embed_dim * 2 , bias=False),
                    nn.Tanh(),
                    Linear(embed_dim * 2, embed_dim * 2, bias=False),
                    nn.Tanh(),
                    Linear(embed_dim * 2, 1, bias=False),
                    # nn.Sigmoid(),   
            )
        else:
            NotImplementedError()

    def forward(
            self,
            sample,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            src_lengths: Optional[Tensor] = None,
    ):
        x, attn_scores = self.extract_features(
            sample, incremental_state
        )
        return x, attn_scores
        # return self.output_layer(x), attn_scores

    def extract_features(
            self,
            sample,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Similar to *forward* but only return features.
        """
        use_feature = False
        if 'src_features' in sample:
            use_feature = True
            assert self.use_feature

        # TOKENS: B X (ACTIONS_SEQUENCE_LEN-1) X T
        if not use_feature:
            tokens = sample['src_tokens']
            prev_output_tokens = sample['prev_output_tokens']
            src_embedding = self.src_embed_tokens(tokens)
            trg_embedding = self.trg_embed_tokens(prev_output_tokens)
        
        else:
            src_embedding = sample['src_features']
            trg_embedding = sample['trg_features']

        src_info = self.src_layers(src_embedding)
        trg_info = self.trg_layers(trg_embedding)
        
        # src_info = torch.sum(src_output,dim=2).squeeze()
        # trg_info = torch.sum(trg_output,dim=2).squeeze()
        
        assert src_info.dim() == trg_info.dim() 
        
        attn_scores = None
        x = {
            'src_info': src_info,
            'trg_info': trg_info,
        }
        return x, attn_scores

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Linear(in_features, out_features, bias=True, dropout=0.0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture("info_linear", "info_linear")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_hidden_size = getattr(
        args, "decoder_hidden_size", args.decoder_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    args.decoder_attention = getattr(args, "decoder_attention", "0")
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "10000,50000,200000"
    )
    args.residuals = getattr(args, "residuals", False)

@register_model_architecture("info_linear", "info_linear_big")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_hidden_size = getattr(
        args, "decoder_hidden_size", args.decoder_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    args.decoder_attention = getattr(args, "decoder_attention", "0")
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "10000,50000,200000"
    )
    args.residuals = getattr(args, "residuals", True)

@register_model_architecture("info_linear", "ainfo_linear_verybig")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_hidden_size = getattr(
        args, "decoder_hidden_size", args.decoder_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 8)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    args.decoder_attention = getattr(args, "decoder_attention", "0")
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "10000,50000,200000"
    )
    args.residuals = getattr(args, "residuals", True)

@register_model_architecture("info_linear", "info_linear_small")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_hidden_size = getattr(
        args, "decoder_hidden_size", args.decoder_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    args.decoder_attention = getattr(args, "decoder_attention", "0")
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "10000,50000,200000"
    )
    args.residuals = getattr(args, "residuals", True)