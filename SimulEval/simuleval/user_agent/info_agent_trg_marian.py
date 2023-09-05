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

from fairseq.data import Dictionary
from simuleval.user_agent.simul_utils import load_policy_model

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
        self.num_write = 0

    def update_target(self, segment: Segment):
        self.target_finished = segment.finished
        if not self.target_finished:
            if isinstance(segment, TextSegment):
                self.target.extend(segment.content.split())
            else:
                self.target += segment.content

@entrypoint
class InfoAgent(TextToTextAgent):
    def __init__(self, args: Namespace):
        """Initialize your agent here.
        For example loading model, vocab, etc
        """
        super().__init__(args)
        self.gpu = getattr(args, "gpu", True)
        self.wait_k = args.wait_k
        self.beam = args.beam_size
        self.temperature = args.temperature

        self.policy_model, self.nmt_model = load_policy_model(args, nmt=True)

        self.tgt_dict = Dictionary.load(args.dict_path)
        self.pad = self.tgt_dict.pad_index
        self.eos = self.tgt_dict.eos_index
        self.pad_token = self.tgt_dict.pad_word
        self.eos_token = self.tgt_dict.eos_word

        self.max_len = args.max_len
        
        self.threshold = args.threshold
        self.agent_arch = args.agent_arch

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--dict-path", type=str)
        parser.add_argument("--policy-model-path", type=str)
        parser.add_argument("--wait-k", type=int, default=1)
        parser.add_argument("--user-dir", type=str)
        parser.add_argument("--max-len", type=int, default=200)
        parser.add_argument("--beam-size", type=int, default=1)
        parser.add_argument("--task", type=str, default='info_policy_task')
        parser.add_argument("--temperature", type=int, default=1)
        parser.add_argument("--threshold", type=float, default=0)
        parser.add_argument("--agent-arch", type=str, default='softplus2')
        
    def build_states(self) -> CustomAgentStates:
        return CustomAgentStates()
        
    def to_device(self, tensor):
        if isinstance(tensor, list):
            return self.to_device(torch.tensor(tensor))
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()
        
    def encode(self, sentences, tgt=False):
        tokens = [self.tgt_dict.index(s) for s in sentences.split()]
        if tgt:
            tokens = tokens + list(self.states.prefix)
        tokens = self.to_device(tokens).unsqueeze(0)
        return tokens
    
    def update_sub_hyp(self, src_indices, tgt_indices):
        if src_indices[0, -1] != self.eos:
            src_indices = torch.concat(
                (src_indices, self.to_device(torch.tensor([[self.eos]]))), -1
            )
        out = self.nmt_model.model.generate(
            input_ids=src_indices,
            decoder_input_ids=tgt_indices,
            return_dict_in_generate=True, 
            num_beams=self.beam,
            num_return_sequences=1
        )
        return out['sequences'][0] # except <pad> for first token

    def cal_info(self, net_output):
        src_info = net_output[0]['src_info']
        trg_info = net_output[0]['trg_info']

        src_info_bias = self.to_device(torch.ones_like(src_info))

        src_info = torch.nn.functional.softplus(src_info)
        if self.agent_arch == 'softplus':
            src_info += src_info_bias
        trg_info = torch.nn.functional.softplus(trg_info)
        
        src_info = src_info.sum()
        trg_info = trg_info.sum()
        valid_info = src_info - trg_info
        
        return valid_info
    
    @torch.no_grad()
    def info_policy(self):
        read_finish = self.states.source_finished

        src_str = ' '.join(self.states.source)
        src_indices = self.encode(src_str)

        tgt_str = ' '.join([self.pad_token] + self.states.target)
        tgt_indices = self.encode(tgt_str)
    
        if read_finish:
            hyp_tokens = self.update_sub_hyp(src_indices, tgt_indices)
            for h in hyp_tokens[tgt_indices.size(-1):]:
                self.states.prefix.append(h.item())
            assert h == self.eos
            return
        
        src_features = self.nmt_model.get_src_features(src_indices)
        while True:
            sub_hyps = self.update_sub_hyp(
                src_indices, tgt_indices
            )[1:] # except <pad> for first token

            sub_idx = self.states.num_write
            if sub_idx > sub_hyps.size(0) - 1:
                cur_trg = self.pad
            else:
                cur_trg = sub_hyps[sub_idx]

            trgs = sub_hyps[:sub_idx + 1].unsqueeze(0)

            trg_features = self.nmt_model.get_trg_features(trgs)

            net_input = {
                'src_features': src_features,
                'trg_features': trg_features,
            }
            info_out = self.policy_model(net_input)
            valid_info =self.cal_info(info_out)

            action = valid_info > self.threshold # True -> Write
            if (trgs[0, -1] == self.pad or 
                trgs[0, -1] == self.eos):
                action = False
            if trgs.size(-1) > 1:
                if trgs[0, -2] == self.eos:
                    action = False

            if action: # write
                self.states.num_write += 1
                self.states.prefix.append(cur_trg.item())
                tgt_indices = self.encode(tgt_str, True)
            else: # read
                return

    def policy(self):
        self.info_policy()
        
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