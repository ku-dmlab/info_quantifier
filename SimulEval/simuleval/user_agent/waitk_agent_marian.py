# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List

from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction, Action
from simuleval.agents.states import AgentStates
from simuleval.data.segments import Segment, EmptySegment, TextSegment, SpeechSegment
from argparse import Namespace, ArgumentParser

from transformers import MarianTokenizer, MarianMTModel
from transformers.generation.logits_process import LogitsProcessorList
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
    
@entrypoint
class WaitKAgent(TextToTextAgent):
    def __init__(self, args: Namespace):
        """Initialize your agent here.
        For example loading model, vocab, etc
        """
        super().__init__(args)
        self.gpu = getattr(args, "gpu", True)
        self.waitk = args.wait_k

        self.tokenizer = MarianTokenizer.from_pretrained(args.hf_name)
        self.model = MarianMTModel.from_pretrained(args.hf_name)
        if self.gpu:
            self.model.cuda()

        self.tgt_dict = Dictionary.load(args.dict_path)
        self.pad = self.tgt_dict.pad_index
        self.eos = self.tgt_dict.eos_index
        self.pad_token = self.tokenizer.pad_token
        self.eos_token = self.tokenizer.eos_token
        self.logits_processor = LogitsProcessorList()

        self.max_len = args.max_len

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--hf-name", type=str, default="Helsinki-NLP/opus-mt-ko-en")
        parser.add_argument("--dict-path", type=str, default="")
        parser.add_argument("--wait-k", type=int, default=3)
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
        # return self.to_device(
        #     self.tokenizer(sentences, return_tensors='pt', padding=True)['input_ids']
        # )
        return tokens
    
    def decode(self, tokens: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    def pushpop(self, source_segment, states = None):
        # push
        self.states.update_source(source_segment)
            
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
    
    @torch.no_grad()
    def sample(self):
        src_str = ' '.join(self.states.source)
        src_indices = self.encode(src_str)
        if self.states.source_finished: # add eos
            src_indices = torch.cat(
                (src_indices, self.to_device([[self.eos]])), -1
            )

        tgt_str = ' '.join([self.pad_token] + self.states.target)
        tgt_indices = self.encode(tgt_str)

        outputs = self.model(
            input_ids=src_indices,
            decoder_input_ids=tgt_indices,
            return_dict=True, 
        )

        next_token_logits = outputs.logits[:, -1, :]
        next_tokens_scores = self.logits_processor(src_indices, next_token_logits)
        hyp_token = torch.argmax(next_tokens_scores, dim=-1)

        finish = False
        if ( hyp_token == self.eos ) or ( tgt_indices.size(1) == self.max_len ):
            finish = True

        return self.tgt_dict.string(hyp_token), finish
            
    def policy(self):
        lagging = len(self.states.source) - len(self.states.target)

        if lagging >= self.waitk or self.states.source_finished:
            pred, finish = self.sample()
            return WriteAction(pred, finished=finish)
        else:
            return ReadAction()
        