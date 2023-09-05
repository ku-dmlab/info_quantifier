# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from collections import deque

import torch
import numpy as np

from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction, Action
from simuleval.agents.states import AgentStates
from simuleval.data.segments import Segment, EmptySegment, TextSegment, SpeechSegment
from argparse import Namespace, ArgumentParser

from simuleval.user_agent.simul_utils import load_mt_model, BeamDecoding

SEGMENT_TYPE_DICT = {"text": TextSegment, "speech": SpeechSegment}

@dataclass
class FinalSegment(Segment):
    detokenized: str = ''
    finished: bool = False

class CustomAgentStates(AgentStates):
    def __init__(self):
        super().__init__()
    
    def reset(self):
        self.source = []
        self.target = []
        self.source_finished = False
        self.target_finished = False
        self.enc_states = None
        self.src_finish = False
        self.detokenized = None
        self.prefix = deque()
        self.beam_hyps = []

    def update_target(self, segment: Segment):
        self.target_finished = segment.finished
        if not self.target_finished:
            if isinstance(segment, TextSegment):
                self.target.extend(segment.content.split())
            else:
                self.target += segment.content

@entrypoint
class LaNAgent(TextToTextAgent):
    def __init__(self, args: Namespace):
        """Initialize your agent here.
        For example loading model, vocab, etc
        """
        super().__init__(args)
        self.gpu = getattr(args, "gpu", True)
        self.wait_k = args.wait_k
        self.la_n = args.la_n
        # assert self.wait_k >= self.la_n
        self.beam_size = args.beam_size
        dictionary = self.load_model_vocab(args.mt_model_path)
        self.src_dict, self.tgt_dict = dictionary
        
        self.pad = self.tgt_dict.pad_index
        self.eos = self.tgt_dict.eos_index

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--mt-model-path", type=str)
        parser.add_argument("--wait-k", type=int, default=3)
        parser.add_argument("--la-n", type=int, default=3)
        parser.add_argument("--beam-size", type=int, default=3)
        
    def build_states(self) -> CustomAgentStates:
        return CustomAgentStates()
        
    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()
        
    def load_model_vocab(self, model_path):
        model, tgt_dict, src_dict = load_mt_model(
            model_path,
            return_dict=True
        )

        self.decode = BeamDecoding(
            models=[model],
            tgt_dict=tgt_dict,
            beam_size=self.beam_size,
            vocab_size=len(tgt_dict),
            tokens_to_suppress=(),
            symbols_to_strip_from_output=None,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=0.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
        )

        if self.gpu:
            self.decode.set_device('cuda')
        else:
            self.decode.set_device('cpu')
            
        return [src_dict, tgt_dict]
        
    def get_tgt_indices(self):
        target = self.states.target

        tgt_indices = self.to_device(
            torch.LongTensor([
                self.tgt_dict.index(x)
                for x in target
                if x is not None
            ])
        ).unsqueeze(0)

        return tgt_indices
    
    def read_finish_decode(self):
        tgt_indices = self.get_tgt_indices()
        hyp_tokens = self.decode.decode(
            [self.states.enc_states],
            tgt_indices,
        )[0][0]['tokens']
        for h in hyp_tokens[tgt_indices.size(-1):]:
            self.states.prefix.append(h.item())
        assert h == self.eos

    def local_agree(self, read_finish):
        if self.states.target_finished:
            return
        
        if read_finish:
            self.read_finish_decode()
            return

        # if not read finish, ignore eos
        last_agree = len(self.states.target)

        if len(self.states.beam_hyps) >= self.la_n:
            n_hypos = self.states.beam_hyps[-self.la_n:]
            comp_hyps =[h[last_agree:] for h in n_hypos]
            for hyps in zip(*comp_hyps):
                t = hyps[0]
                if all(o == t for o in hyps) and t != self.eos:
                    self.states.prefix.append(t.item())
                else:
                    return

    def policy(self):
        read_finish = self.update_src()

        self.local_agree(read_finish)
        
        if self.states.prefix:
            pred, finish = self.prefix()
            return WriteAction(pred, finished=finish)
    
        return ReadAction()
    
    @torch.no_grad()
    def update_src(self):
        if self.states.src_finish:
            return True

        read_finish = self.states.source_finished
        
        src_indices = self.to_device(
            torch.LongTensor([
                self.src_dict.index(x) 
                for x in self.states.source
            ] + [self.eos]) # eos for beam search
        ).unsqueeze(0)
        src_lengths = self.to_device(
            torch.LongTensor([src_indices.size(1)])
        )
        encoder = self.decode.model.single_model.encoder
        enc_out = encoder(src_indices, src_lengths)
        self.states.enc_states = enc_out

        lag = len(self.states.source) - len(self.states.target)
        if lag > self.wait_k:
        # if lag > (self.wait_k - self.la_n):
            tgt_indices = self.get_tgt_indices()
            beam_hyps = self.decode.decode(
                [enc_out], tgt_indices
            )[0][0]['tokens']
            self.states.beam_hyps.append(beam_hyps)
            if tgt_indices.size(-1) > 0:
                assert (tgt_indices[0] == beam_hyps[:tgt_indices.size(-1)]).all()
        
        if read_finish:
            self.states.src_finish = True

        return read_finish
    
    def prefix(self):
        tokens = []
        while self.states.prefix:
            index = self.states.prefix.popleft()
            if index != self.eos:
                tokens.append(
                    self.tgt_dict.string([int(index)])
                )
            else:
                if tokens:
                    return ' '.join(tokens), True
                else:
                    return '', True

        return ' '.join(tokens), False