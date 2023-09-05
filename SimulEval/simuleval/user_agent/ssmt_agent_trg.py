# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# TRG -> decoding token

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

from simuleval.user_agent.simul_utils import load_mt_model, load_policy_model, BeamDecoding

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
        self.prefix = deque()
        
        self.src_inds = None
        self.subsets = []
        self.prev_actions = []
        self.num_write = 0
        self.incremental_states = {}

    def update_target(self, segment: Segment):
        self.target_finished = segment.finished
        if not self.target_finished:
            if isinstance(segment, TextSegment):
                self.target.extend(segment.content.split())
            else:
                self.target += segment.content

@entrypoint
class SSMTAgent(TextToTextAgent):
    def __init__(self, args: Namespace):
        """Initialize your agent here.
        For example loading model, vocab, etc
        """
        super().__init__(args)
        self.gpu = getattr(args, "gpu", True)
        self.wait_k = args.wait_k
        self.beam_size = args.beam_size
        self.temperature = args.temperature

        self.policy_model = load_policy_model(args)
        dictionary = self.load_model_vocab(args.mt_model_path)
        self.src_dict, self.tgt_dict = dictionary
        self.mt_encoder = self.decode.model.single_model.encoder
        self.mt_decoder = self.decode.model.single_model.decoder
        
        self.pad = self.tgt_dict.pad_index
        self.eos = self.tgt_dict.eos_index
        
        self.no_prefix = args.no_prefix

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--mt-model-path", type=str)
        parser.add_argument("--policy-model-path", type=str)
        parser.add_argument("--wait-k", type=int, default=1)
        parser.add_argument("--user-dir", type=str)
        parser.add_argument("--data-bin", type=str)
        parser.add_argument("--max-len", type=int, default=200)
        parser.add_argument("--beam-size", type=int, default=3)
        parser.add_argument("--task", type=str, default='ssmt_task')
        parser.add_argument("--no-prefix", type=bool, default=False)
        parser.add_argument("--temperature", type=int, default=1)
        
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
            ] + list(self.states.prefix))
        ).unsqueeze(0)
        # assert tgt_indices.size(-1) == self.states.num_write
        return tgt_indices
    
    def read_finish_decode(self):
        enc_out = self.mt_encoder(self.states.src_inds.unsqueeze(0))
        tgt_indices = self.get_tgt_indices()
        hyp_tokens = self.decode.decode([enc_out], tgt_indices)[0][0]['tokens']
        for h in hyp_tokens[tgt_indices.size(-1):]:
            self.states.prefix.append(h)
    
    def update_sub_hyp(self):
        src_indices = self.states.src_inds.clone()
        src_indices = torch.cat(
            (src_indices, self.to_device(torch.tensor([self.eos]))), -1
        )
        src_indices = src_indices.unsqueeze(0)
        enc_out = self.mt_encoder(src_indices)
        prev_indices = self.get_tgt_indices()
        sub_hyps = self.decode.decode([enc_out], prev_indices)

        return sub_hyps[0][0]['tokens']

    @torch.no_grad()
    def ssmt_policy(self, read_finish):
        if self.states.target_finished:
            return
    
        if read_finish:
            self.read_finish_decode()
            return
        
        while True:
            sub_hyps = self.update_sub_hyp()
            cur_src = self.states.src_inds[-1]
            sub_idx = self.states.num_write
            if sub_idx > sub_hyps.size(0) - 1:
                cur_trg = self.pad
            else:
                cur_trg = sub_hyps[sub_idx]

            self.states.subsets.append(
                self.to_device(torch.tensor([cur_src, cur_trg]))
            )
            ssmt_out = self.policy_model(
                torch.stack(self.states.subsets).unsqueeze(0),
                torch.stack(self.states.prev_actions).unsqueeze(0), 
                incremental_state=self.states.incremental_states,
            )
            lprobs = self.policy_model.get_normalized_probs(ssmt_out, log_probs=True)
            lprobs[:, :, :4] = -torch.inf
            # constraint 1: cur src is eos, no read
            if cur_src == self.eos:
                lprobs[:, :, 4] = -torch.inf
            # constraint 2: cur trg is pad, no write
            if cur_trg == self.pad:
                lprobs[:, :, 5] = -torch.inf
            # constraint 3: cur trg is eos & action write -> read
            if cur_trg == self.eos:
                lprobs[:, :, 5] = -torch.inf
            # constraint 3: prev trg is eos, no write
            if len(self.states.subsets) > 1:
                if self.states.subsets[-2][1] == self.eos:
                    lprobs[:, :, 5] = -torch.inf

            action = lprobs.argmax(-1)[0, -1].item()

            prev_act = self.states.prev_actions[-1][1:].clone()
            cur_act = torch.concat(
                (prev_act, torch.tensor([action]).to(prev_act)), -1
            )
            if action == 4:
                self.states.prev_actions.append(cur_act)
                return
            else: # write
                self.states.num_write += 1
                self.states.prev_actions.append(cur_act)
                self.states.prefix.append(cur_trg)

    def policy(self):
        read_finish = self.update_src()
        
        self.ssmt_policy(read_finish)
        if self.states.prefix:
            pred, finish = self.prefix()
            return WriteAction(pred, finished=finish)
        
        return ReadAction()
    
    @torch.no_grad()
    def update_src(self):
        if (self.states.src_inds is not None and 
            self.states.src_inds[-1] == self.eos
        ):
            return True

        read_finish = self.states.source_finished
        
        src_indices = self.to_device(
            torch.LongTensor([
                self.src_dict.index(x) 
                for x in self.states.source
            ] + [self.eos]) # eos for beam search
        ).unsqueeze(0)
        
        self.states.src_inds = src_indices[0]
        if not read_finish:
            self.states.src_inds = src_indices[0, :-1]

        if not self.states.prev_actions:
            self.states.prev_actions.append(
                self.to_device(torch.tensor([1,1,4]))
            )

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