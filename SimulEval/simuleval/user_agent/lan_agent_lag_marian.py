# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from collections import deque

import torch

from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction, Action
from simuleval.agents.states import AgentStates
from simuleval.data.segments import Segment, EmptySegment, TextSegment, SpeechSegment
from argparse import Namespace, ArgumentParser

from transformers import MarianTokenizer, MarianMTModel
from fairseq.data import Dictionary

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
        super().__init__(args)
        self.gpu = getattr(args, "gpu", True)
        self.wait_k = args.wait_k
        self.la_n = args.la_n
        self.beam = args.beam_size

        self.tokenizer = MarianTokenizer.from_pretrained(args.hf_name)
        self.model = MarianMTModel.from_pretrained(args.hf_name)
        if self.gpu:
            self.model.cuda()

        self.tgt_dict = Dictionary.load(args.dict_path)
        self.pad = self.tgt_dict.pad_index
        self.eos = self.tgt_dict.eos_index
        self.pad_token = self.tokenizer.pad_token
        self.eos_token = self.tokenizer.eos_token

        self.max_len = args.max_len

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--hf-name", type=str, default="Helsinki-NLP/opus-mt-ko-en")
        parser.add_argument("--dict-path", type=str, default="")
        parser.add_argument("--wait-k", type=int, default=3)
        parser.add_argument("--la-n", type=int, default=3)
        parser.add_argument("--beam-size", type=int, default=3)
        parser.add_argument("--max-len", type=int, default=200)
        
    def build_states(self) -> CustomAgentStates:
        return CustomAgentStates()
        
    def to_device(self, tensor):
        if isinstance(tensor, list):
            return self.to_device(torch.tensor(tensor))
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()
        
    def encode(self, sentences):
        tokens = [self.tgt_dict.index(s) for s in sentences.split()]
        tokens = self.to_device(tokens).unsqueeze(0)
        return tokens
    
    def read_finish_decode(self, src_ids, tgt_ids):
        out = self.model.generate(
            input_ids=src_ids,
            decoder_input_ids=tgt_ids,
            return_dict_in_generate=True, 
            num_beams=self.beam,
            num_return_sequences=1
        )
        hyp_tokens = out['sequences'][0]
        for h in hyp_tokens[tgt_ids.size(-1):]:
            self.states.prefix.append(h.item())
        assert h == self.eos

    @torch.no_grad()
    def local_agree(self, lag):
        read_finish = self.states.source_finished

        src_str = ' '.join(self.states.source + [self.eos_token])
        src_indices = self.encode(src_str)

        tgt_str = ' '.join([self.pad_token] + self.states.target)
        tgt_indices = self.encode(tgt_str)

        if read_finish: # add eos
            self.read_finish_decode(src_indices, tgt_indices)
            return
        
        out = self.model.generate(
            input_ids=src_indices,
            decoder_input_ids=tgt_indices,
            return_dict_in_generate=True, 
            num_beams=self.beam,
            num_return_sequences=1
        )
        # if not read finish, ignore eos
        last_agree = len(self.states.target)
        beam_hyps = out['sequences'][0]
        if tgt_indices.size(-1) > 0:
            assert (tgt_indices[0] == beam_hyps[:tgt_indices.size(-1)]).all()
        self.states.beam_hyps.append(beam_hyps[1:]) # except <pad> for first token

        if len(self.states.beam_hyps) >= self.la_n and lag > self.wait_k:
            n_hypos = self.states.beam_hyps[-self.la_n:]
            comp_hyps =[h[last_agree:] for h in n_hypos]
            for hyps in zip(*comp_hyps):
                t = hyps[0]
                if all(o == t for o in hyps) and t != self.eos:
                    self.states.prefix.append(t.item())
                else:
                    return

    def policy(self):
        lag = len(self.states.source) - len(self.states.target)
        self.local_agree(lag)

        if self.states.prefix:
            pred, finish = self.prefix()
            return WriteAction(pred, finished=finish)
    
        return ReadAction()
    
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