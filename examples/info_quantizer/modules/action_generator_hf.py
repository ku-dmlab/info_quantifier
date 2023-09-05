# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import search, utils
from torch import Tensor

class HfActionGenerator(nn.Module):
    def __init__(
            self,
            models,
            tgt_dict,
            src_dict,
            beam_size=1,
            max_len_a=0,
            max_len_b=200,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=0.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
            search_strategy=None,
            eos=None,
            symbols_to_strip_from_output=None,
            has_target=True,
            teacher_forcing_rate=1.,
            distort_action_sequence=False,
            full_read=False,
            **kwargs,
    ):
        super().__init__()
        self.model = models[0]
        self.tgt_dict = tgt_dict
        self.src_dict = src_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size

        # If True then _generate() returns Encoder and Decoder outputs as well.
        # self.simple_models = ['agent_lstm', 'agent_lstm_0', 'agent_lstm_big', 'agent_lstm_0_big', None]
        # self.all_features = False if agent_arch in self.simple_models else True
        self.all_features = True
        self.has_target = has_target
        self.teacher_forcing_rate = teacher_forcing_rate
        self.distort_action_sequence = distort_action_sequence

        assert temperature > 0, "--temperature must be greater than 0"

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
                hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
        )
        self.model.eval()
        self.full_read=full_read

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    # TODO(myleott): unused, deprecate after pytorch-translate migration
    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if "net_input" not in s:
                continue
            input = s["net_input"]
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items() if k != "prev_output_tokens"
            }
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]["tokens"]) for h in hypos))
            for i, id in enumerate(s["id"].data):
                # remove padding
                src = utils.strip_pad(input["src_tokens"].data[i, :], self.pad)
                ref = (
                    utils.strip_pad(s["target"].data[i, :], self.pad)
                    if s["target"] is not None
                    else None
                )
                yield id, src, ref, hypos[i]

    def extract_incremental_samples(self, sample):
        incremental_sample = {
            k: v for k, v in sample.items() if k != "net_input"
        }
        new_net_input = []
        sample_net_input = sample['net_input']

        # If the first element of a sample is pad, replace it with unk. (We'll get errors when generating translations)
        first = sample_net_input['src_tokens'][:, 0]
        if self.pad in first:
            for batch_index in range(sample_net_input['src_tokens'].shape[0]):
                if sample_net_input['src_tokens'][batch_index, 0] == self.pad:
                    sample_net_input['src_tokens'][batch_index, 0] = self.unk

        max_len = sample_net_input['src_tokens'].shape[1]
        for i in range(max_len):
            num_undone = (sample_net_input['src_lengths'] > i).sum()
            source_tokens = sample_net_input['src_tokens'][:num_undone, : i+1]

            # Add EOS to unfinished sentences
            last_elements = torch.tensor([self.eos if element[-1] not in [self.eos, self.pad] else self.pad
                                          for element in source_tokens], device=source_tokens.device)
            source_tokens = torch.cat((source_tokens, torch.unsqueeze(last_elements, 1)), 1)
            new_net_input.append(
                {'src_tokens': source_tokens,
                 'src_lengths': torch.min(sample_net_input['src_lengths'][:num_undone],
                                          torch.tensor([i + 2] * num_undone, device=source_tokens.device)),}
                #  'full_src_lengths': sample_net_input['src_lengths'][:num_undone]
                #      }  # full_src_lengths will be used in _generate_train() to determine the length of target sentences.
            )
        incremental_sample['net_input'] = new_net_input
        return incremental_sample

    def prepare_post_process(self, src_token):
        src_str = None
        src_tokens = utils.strip_pad(
            src_token, self.tgt_dict.pad()
        )
        if self.src_dict is not None:
            src_str = self.src_dict.string(src_tokens)

        return src_str

    def extend_sample(self, translations, sample):
        net_input = sample['net_input']
        action_seq = sample['action_seq']
        extended_translation = []
        for index, translation in enumerate(translations):
            subsets_generated = [subset[0] for subset in translation]
            extended_sample = translation[-1][0]
            extended_sample['src_tokens'] = net_input['src_tokens'][index]
            extended_sample['action_seq'] = action_seq[index]
            extended_sample['subsets'] = subsets_generated

            # Generating translations based on action sequences.
            generateed_translation = []
            i, j = 0, 0
            read, write = 4, 5
            for action in action_seq[index]:
                if action == read:
                    j = j + 1
                else:
                    generateed_translation.append(
                        subsets_generated[j-1]['tokens'][i] if i < len(subsets_generated[j-1]['tokens'])
                        else subsets_generated[j-1]['tokens'][-2]  # The last element is the eos token. We take the one before.
                    )
                    i = i + 1

            extended_sample['tokens'] = generateed_translation
            extended_translation.append(extended_sample)
        return extended_translation

    def generate_action_sequence(self, translations, net_input=None, post_process=True):
        extended_translations = []
        src_tokens = net_input['src_tokens']
        src_lengths = net_input['src_lengths']
        for index, translation in enumerate(translations):
            if self.full_read:
                extended_translations.append(
                    self._generate_action_sequence_full_read(translation, src_tokens[index],
                    src_lengths[index], post_process)
                )
            else:
                extended_translations.append(
                    self._generate_action_sequence(
                        translation, src_tokens[index], src_lengths[index]
                    )
                )
        return extended_translations

    def _generate_action_sequence_full_read(self, sample, src_token, src_length, post_process):
        i, j = 0, 0 # Write, Read
        action_seq = [0]
        trg_Gen = sample[-1][0]['tokens']

        trg_length = len(trg_Gen)

        # Generate action based on the first beam.
        subsets_generated = [subset[0] for subset in sample]

        if post_process:
            src_str = self.prepare_post_process(src_token)
            for index, sub_gen in enumerate(subsets_generated):
                _, hypo_str, _ = utils.post_process_prediction(
                    hypo_tokens=sub_gen["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=sub_gen["alignment"],
                    align_dict=None,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=None,
                    extra_symbols_to_ignore=None,
                )
                hypo_str = hypo_str.strip().split(' ') + ['<EOS>']
                subsets_generated[index]["hypo_str"] = hypo_str
            trg_Gen = hypo_str
            trg_length = len(trg_Gen)

        while i < trg_length:
            subset = subsets_generated[j]['hypo_str'] if post_process else subsets_generated[j]['tokens']
            if i < len(subset) and subset[i] == trg_Gen[i]:
                action_seq.append(1)
                i = i + 1 # Write
            elif j + 1 < src_length:
                if j + 3 == src_length:     # The last two translations are the same
                    action_seq.append(0)
                    j = j + 1 # Read
                action_seq.append(0)
                j = j + 1 # Read

        # mk
        add_read = (src_length.item() - 1) - j
        if add_read != 0: # Not Read finish
            last_act = action_seq[-1:]
            action_seq = action_seq[:-1] + [0 for _ in range(add_read)] + last_act

        # # Debug
        # if j + 1 < src_length:
        #     # print("WARNING --> Some zeros added")
        #     while j + 1 < src_length:
        #         action_seq.append(0)
        #         j = j + 1

        # assert (src_length + trg_length == len(action_seq)), [src_length, trg_length, len(action_seq)]
        extended_sample = sample[-1][0]
        extended_sample['src_tokens'] = src_token
        extended_sample['action_seq'] = action_seq
        extended_sample['subsets'] = subsets_generated
        return extended_sample
    
    def _generate_action_sequence(self, sample, src_token, src_length):
        i, j = 0, 0 # Write, Read
        action_seq = [0]
        trg_Gen = sample[-1][0]['tokens']

        trg_length = len(trg_Gen)

        # Generate action based on the first beam.
        subsets_generated = [subset[0] for subset in sample]

        while i < trg_length:
            subset = subsets_generated[j]['tokens']
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # if i < len(subset) and subset[i] == trg_Gen[i]:
            if i < len(subset) and subset[i] in trg_Gen:
                action_seq.append(1)
                i = i + 1 # Write
            elif j + 1 < src_length:
                if j + 3 == src_length:     # The last two translations are the same
                    action_seq.append(0)
                    j = j + 1 # Read
                action_seq.append(0)
                j = j + 1 # Read

        # # Debug
        # if j + 1 < src_length:
        #     # print("WARNING --> Some zeros added")
        #     while j + 1 < src_length:
        #         action_seq.append(0)
        #         j = j + 1

        # assert (src_length + trg_length == len(action_seq)), [src_length, trg_length, len(action_seq)]
        extended_sample = sample[-1][0]
        extended_sample['src_tokens'] = src_token
        extended_sample['action_seq'] = action_seq
        extended_sample['subsets'] = subsets_generated
        return extended_sample


    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        incremental_samples = self.extract_incremental_samples(sample)
        for i in range(sample['net_input']['src_tokens'].shape[1]):
            subsample = {
                k: v for k, v in sample.items() if k != "net_input"
            }
            subsample['net_input'] = incremental_samples['net_input'][i]

            partial_translation = self._generate(subsample)

            if i == 0:
                translations = [[partial] for partial in partial_translation]
            else:
                for index, trans in enumerate(partial_translation):
                    translations[index].append(trans)
        
        if sample['action_seq'] is not None:
            return self.extend_sample(translations, sample)
        return self.generate_action_sequence(translations, sample['net_input'])

    def _generate(self, sample: Dict[str, Dict[str, Tensor]]):
        input_ids = sample['net_input']['src_tokens']
        gen_out = self.model.generate(
            input_ids,
            num_beams=self.beam_size,
            num_return_sequences=1,
        )

        def padding(toks, feats, pad):
            b, t, d = feats.size()
            mask = (toks == pad)
            mask = mask.unsqueeze(-1).repeat(1, 1, d)
            feats[mask] = 0.
            padded = torch.zeros(b, 200, d).to(feats)
            padded[:, :t] = feats
            return padded
        
        tokens = gen_out['sequences'][:, 1:]
        # src_features = gen_out['encoder_hidden_states'][-1] # B x T x D
        # trg_features = self.model.get_trg_features(tokens) # B X T x D

        # padded_src_feats = padding(input_ids, src_features, self.pad)
        # padded_trg_feats = padding(tokens, trg_features, self.pad)
        
        finalized = self.refine_hypos(
            tokens=tokens,
            src_tokens=input_ids,
            # src_features=padded_src_feats,
            # trg_features=padded_trg_feats
        )
        return finalized

    def refine_hypos(
            self,
            tokens,
            src_tokens,
            # src_features: Optional[Tensor],
    ):
        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(len(tokens))],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # For every finished beam item
        for i in range(len(tokens)):
            # Remove all tokens generated after the first EOS token.
            current_token = tokens[i]
            if current_token.dim() == 0:
                current_token = torch.unsqueeze(current_token, dim=0)
            eos_idx = torch.nonzero(current_token==self.eos)
            if len(eos_idx) > 0:
                final_tokens = current_token[:int(eos_idx[0])+1]
            else:
                final_tokens = torch.cat(
                    (current_token, torch.tensor([self.eos]).to(tokens))
                ).to(tokens)

            finalized[i].append(
                {
                    "tokens": final_tokens,
                    "src_tokens": src_tokens[i],
                    "score": torch.tensor(0.0).type(torch.float),
                    "attention": torch.empty(0), # src_len x tgt_len
                    # "src_features": src_features[i],
                    "alignment": torch.empty(0),
                    "positional_scores": torch.tensor(0.0).type(torch.float),
                }
            # score, alignment, positional_scores are there to keep output format
            )
        return finalized


