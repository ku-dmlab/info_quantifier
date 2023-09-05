# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from dataclasses import dataclass

from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction, Action
from simuleval.agents.states import AgentStates
from simuleval.data.segments import Segment, EmptySegment, TextSegment, SpeechSegment
from argparse import Namespace, ArgumentParser

from simuleval.user_agent.simul_utils import load_mt_model

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
        self.enc_fix = False
        self.detokenized = None
    
@entrypoint
class WaitKAgent(TextToTextAgent):
    def __init__(self, args: Namespace):
        """Initialize your agent here.
        For example loading model, vocab, etc
        """
        super().__init__(args)
        self.gpu = getattr(args, "gpu", True)
        self.waitk = args.wait_k
        # self.waitk = torch.inf
        self.mt_model, dictionary = self.load_model_vocab(args.mt_model_path)
        self.src_dict, self.tgt_dict = dictionary
        
        self.pad = self.tgt_dict.pad_index
        self.eos = self.tgt_dict.eos_index

        self.max_len = args.max_len

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--mt-model-path", type=str)
        parser.add_argument("--wait-k", type=int, default=3)
        parser.add_argument("--max-len", type=int, default=200)
        
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
        if self.gpu:
            model.cuda()
            
        return model, [src_dict, tgt_dict]
    
    def pushpop(self, source_segment, states = None):
        # push
        self.states.update_source(source_segment)
        
        if not self.states.enc_fix:
            src_indices = self.to_device(
                torch.LongTensor(
                    [self.src_dict.index(x)
                    for x in self.states.source]
                )
            ).unsqueeze(0)
            
            if self.states.source_finished:
                src_indices = torch.concat(
                    (src_indices, self.to_device(torch.tensor([[self.eos]]))),
                    -1
                )
                self.states.enc_fix = True

            src_lengths = self.to_device(
                torch.LongTensor([src_indices.size(1)])
            )
            self.states.enc_states = self.mt_model.encoder(
                src_indices, src_lengths
            )
            
        # pop
        if self.states.target_finished:
            return FinalSegment(finished=True)

        action = self.policy()

        if not isinstance(action, Action):
            raise RuntimeError(
                f"The return value of {self.policy.__qualname__} is not an {Action.__qualname__} instance"
            )
        if action.is_read():
            return EmptySegment()
        else:
            if isinstance(action.content, Segment):
                return action.content

            segment = SEGMENT_TYPE_DICT[self.target_type](
                index=0, content=action.content, finished=action.finished
            )
            self.states.update_target(segment)
            return segment
        
    def decode(self):
        tgt_indices = self.to_device(
            torch.LongTensor(
                [self.eos] + 
                [self.tgt_dict.index(x)
                    for x in self.states.target
                    if x is not None]
                # + list(self.states.prefix)
            )
        ).unsqueeze(0)
        out = self.mt_model.decoder(
            prev_output_tokens=tgt_indices,
            encoder_out=self.states.enc_states,
        )
        lprobs = self.mt_model.get_normalized_probs(
            [out[0][:, -1:]], log_probs=True
        )
        hypo = lprobs.argmax(dim=-1)
        
        token = self.tgt_dict.string(hypo)
        
        finish = False
        if (hypo[0, 0].item() == self.eos or tgt_indices.size(1) == self.max_len):
            finish = True

        return token, finish
            
    def policy(self):
        lagging = len(self.states.source) - len(self.states.target)

        if lagging >= self.waitk or self.states.source_finished:
            pred, finish = self.decode()
            return WriteAction(pred, finished=finish)
        else:
            return ReadAction()
        