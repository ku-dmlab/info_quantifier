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
        
        self.sub_src = None
        self.sub_hyp = None
        self.write_hyp = []
        self.subsets = []
        self.num_write = 0
        self.enc_feature = None

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
        self.beam_size = args.beam_size
        self.temperature = args.temperature

        self.policy_model = load_policy_model(args)
        dictionary = self.load_model_vocab(args.mt_model_path)
        self.src_dict, self.tgt_dict = dictionary
        self.mt_encoder = self.decode.model.single_model.encoder
        self.mt_decoder = self.decode.model.single_model.decoder
        
        self.pad = self.tgt_dict.pad_index
        self.eos = self.tgt_dict.eos_index
        
        self.threshold = args.threshold

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
        parser.add_argument("--temperature", type=int, default=1)
        parser.add_argument("--threshold", type=float, default=0)
        
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

        assert tgt_indices.size(-1) == self.states.num_write
        return tgt_indices
    
    def read_finish_decode(self):
        tgt_indices = self.get_tgt_indices()
        hyp_tokens = self.decode.decode(
            [self.states.enc_states],
            tgt_indices,
        )[0][0]['tokens']

        for h in hyp_tokens[tgt_indices.size(-1):]:
            self.states.prefix.append(h)
        assert h == self.eos

    def cal_info(self, net_output):
        src_info = net_output[0]['src_info']
        trg_info = net_output[0]['trg_info']

        src_info_bias = ~(src_info == 0)
        trg_info_bias = ~(trg_info == 0)

        src_info = torch.nn.functional.softplus(src_info) #+ src_info_bias
        trg_info = torch.nn.functional.softplus(trg_info) #+ trg_info_bias
        
        src_info = src_info.sum()
        trg_info = trg_info.sum()
        valid_info = src_info - trg_info
        
        return valid_info
    

    @torch.no_grad()
    def info_policy(self, read_finish):
        if self.states.target_finished:
            return
    
        if read_finish:
            self.read_finish_decode()
            return
        
        while True:
            sub_idx = self.states.num_write

            if sub_idx > self.states.sub_hyp.size(0) - 1:
                cur_trg = self.pad
            else:
                cur_trg = self.states.sub_hyp[sub_idx]

            trgs = self.to_device(torch.tensor(
                [self.states.write_hyp + [cur_trg]]
            ).long())
            dec_features = self.mt_decoder(
                encoder_out=None,
                features_only=True,
                prev_output_tokens=trgs,
            )[0]

            net_input = {
                'src_features': self.states.enc_feature,
                'trg_features': dec_features,
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
                tgt_indices = self.get_tgt_indices()
                hyp_token = self.decode.one_step_decode(
                        [self.states.enc_states],
                        tgt_indices,
                    )
                # hyp_tokens = self.decode.decode(
                #     [self.states.enc_states],
                #     tgt_indices,
                # )[0][0]['tokens']
                # hyp_token = hyp_tokens[tgt_indices.size(-1)]

                # if hyp_token == self.eos:
                #     return 
                # else:
                #     self.states.num_write += 1
                #     self.states.write_hyp.append(cur_trg)
                #     self.states.prefix.append(hyp_token)
                self.states.num_write += 1
                self.states.write_hyp.append(cur_trg)
                self.states.prefix.append(hyp_token)
            
            else: # read
                return

    def policy(self):
        read_finish = self.update_src()
        
        self.info_policy(read_finish)
        if self.states.prefix:
            pred, finish = self.prefix()
            return WriteAction(pred, finished=finish)
        
        return ReadAction()
    
    @torch.no_grad()
    def update_src(self):
        if (self.states.sub_src is not None and 
            self.states.sub_src[-1] == self.eos
        ):
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
        enc_out = self.mt_encoder(src_indices, src_lengths)
        sub_hyps = self.decode.decode([enc_out])
        
        self.states.sub_hyp = sub_hyps[0][0]['tokens']
        self.states.sub_src = src_indices[0]
        self.states.enc_states = enc_out

        if not read_finish:
            self.states.sub_src = src_indices[0, :-1]
            self.states.enc_states = self.mt_encoder(
                src_indices[:, :-1], src_lengths - 1
            )
            src_features = self.mt_encoder(
                src_indices[:, :-1], src_lengths - 1
            )['encoder_out'][0].transpose(1,0).contiguous() 
            self.states.enc_feature = src_features

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