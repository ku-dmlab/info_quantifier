import torch
from examples.info_quantizer.modules.marian_interface import MarianInterface

@torch.no_grad()
def modify_preprocessed_input(sample, task, use_transformer_feature=False, nmt_model=None):

    if task.cfg.arch.startswith("info"):
        pad_index = task.target_dictionary.pad_index
        eos_index = task.target_dictionary.eos_index
        unk_index = task.target_dictionary.unk_index

        if use_transformer_feature:
            bsz, seq_len = sample['action_seq'][:, 1:].size()
            device = sample['target'].device
            nmt_model.to(device)

            cum_srcs = sample['net_input']['src_tokens'].clone()
            cum_trgs = sample['target'].clone()

            src_tokens = torch.full((bsz, seq_len), pad_index).to(device)
            trg_tokens = torch.full((bsz, seq_len), pad_index).to(device)

            read_nums = torch.full((bsz, seq_len), -1).to(device)
            write_nums = torch.full((bsz, seq_len), -1).to(device)

            for idx, act in enumerate(sample['action_seq']):
                act = act[act.eq(4) | act.eq(5)]
                act[act.eq(4)] = 1
                act[act.eq(5)] = 0
                act_len = act.size(-1) - 1
                
                raw_src = cum_srcs[idx, :act_len][act[:-1] == 1]
                src_tokens[idx, :raw_src.size(-1)] = raw_src

                raw_trg = cum_trgs[idx, :act_len][act[1:] == 0]
                trg_tokens[idx, :raw_trg.size(-1)] = raw_trg
                
                cum_read = act[:-1].cumsum(-1) - 1
                cum_read[cum_read < 0] = 0
                read_nums[idx, :cum_read.size(-1)] = cum_read

                cum_write = (~(act[1:].bool())).long().cumsum(-1) - 1
                cum_write[cum_write < 0] = 0
                write_nums[idx, :cum_write.size(-1)] = cum_write

            mask = torch.arange(seq_len).unsqueeze(0).to(device)
            mask = mask.repeat(seq_len * bsz, 1).view(bsz, seq_len, -1)
            read_mask = read_nums.unsqueeze(-1).repeat(1, 1, seq_len)
            write_mask = write_nums.unsqueeze(-1).repeat(1, 1, seq_len)

            src_tokens = src_tokens.unsqueeze(1).repeat_interleave(seq_len, dim=1)
            src_tokens.masked_fill_(mask > read_mask, pad_index)

            trg_tokens = trg_tokens.unsqueeze(1).repeat_interleave(seq_len, dim=1)
            trg_tokens.masked_fill_(mask > write_mask, pad_index)

            read_token_idx = (trg_tokens != 1).sum(-1).unsqueeze(-1)
            read_tokens = sample['target'].clone().unsqueeze(-1)
            read_token_mask = (sample['action_seq'][:, 1:] == 4).unsqueeze(-1)
            read_tokens[~read_token_mask] = 1
            trg_tokens = torch.cat(
                (trg_tokens, torch.full(trg_tokens[:, :, :1].size(), pad_index).to(trg_tokens)), -1
            )
            trg_tokens.scatter_(-1, read_token_idx, read_tokens)
            trg_tokens = trg_tokens[:, :, :-1]

            src_tokens = src_tokens.view(-1, seq_len)
            trg_tokens = trg_tokens.view(-1, seq_len)

            # pad_mask = (sample['action_seq'][:, 1:].flatten() != 1)
            valid_mask = src_tokens[:, 0].ne(pad_index)

            if task.except_eos:
                eos_mask = src_tokens.eq(eos_index).any(-1) | trg_tokens.eq(eos_index).any(-1)
                valid_mask = valid_mask & (~eos_mask)

            src_lenghts = (src_tokens != pad_index).sum(-1)
            src_lenghts = src_lenghts[valid_mask]
            trg_lenghts = (trg_tokens != pad_index).sum(-1)
            trg_lenghts = trg_lenghts[valid_mask]

            src_tokens = src_tokens[valid_mask]
            trg_tokens = trg_tokens[valid_mask]
            if isinstance(nmt_model, MarianInterface):
                src_features = nmt_model.get_src_features(src_tokens)
                trg_features = nmt_model.get_trg_features(src_tokens)
            else:
                enc_outs = nmt_model.encoder(
                    src_tokens,
                    src_lenghts,
                )
                src_features = enc_outs['encoder_out'][0] # L x B * L x dim(512)
                src_features = src_features.transpose(1,0).contiguous()

                trg_features = nmt_model.decoder(
                    encoder_out=None,
                    features_only=True,
                    prev_output_tokens=trg_tokens,
                )[0] # B * L x L * dim(512)
            
            assert torch.isnan(src_features).sum() == 0
            assert torch.isnan(trg_features).sum() == 0

            actions = sample['action_seq'][:, 1:].flatten()[valid_mask]
            net_input = {
                'src_tokens': src_tokens,
                'trg_tokens': trg_tokens,
                'src_features': src_features,
                'trg_features': trg_features,
                'src_lengths': src_lenghts,
                'trg_lengths': trg_lenghts,
                'prev_output_tokens': None,
            }
            new_sample = {
                'id': sample['id'],
                'nsentences': (sample['nsentences'] * seq_len) - (~valid_mask).sum(),
                'ntokens': src_lenghts.sum(-1),
                'net_input': net_input,
                'target': actions,
            }
            return new_sample

            # bsz = sample['nsentences']
            # device = sample['target'].device
            # nmt_model.to(device)
            # src_lens = (sample['action_seq'][:, :-1] == 4).sum(-1)
            # trg_lens = (sample['action_seq'][:, :-1] == 5).sum(-1)
            # max_src_len, max_trg_len = src_lens.max(), trg_lens.max()

            # src_tokens = torch.full((bsz, max_src_len), pad_index).to(device).long()
            # trg_tokens = torch.full((bsz, max_trg_len), pad_index).to(device).long()
            # read_inds = torch.full((bsz, max_trg_len), -1).to(device).long()

            # for index, action_seq in enumerate(sample['action_seq']):
            #     act = action_seq.clone()[:-1]
            #     act = act[act.eq(4) | act.eq(5)]
            #     act[act == 4] = 1
            #     act[act == 5] = 0

            #     raw_src = sample['net_input']['src_tokens'][index, :act.size(-1)][act == 1]
            #     src_tokens[index, :raw_src.size(-1)] = raw_src

            #     cum_read_idx = (act.cumsum(-1) - 1)[act == 0]
            #     read_inds[index, :cum_read_idx.size(-1)] = cum_read_idx
                
            #     raw_trg = sample['target'][index, :act.size(-1)][act == 0]
            #     trg_tokens[index, :raw_trg.size(-1)] = raw_trg

            # enc_features = nmt_model.encoder(
            #     src_tokens, src_lens
            # )['encoder_out'][0] # ST x B x dim(512)
            # enc_features = enc_features.transpose(1,0).contiguous() 
            # ind = read_inds.unsqueeze(-1).repeat(1, 1, enc_features.size(-1))
            # src_mask = (ind == -1)
            # ind[ind == -1] = 0
            # src_features = torch.gather(enc_features, dim=1, index=ind)
            # src_features[src_mask] = 1.

            # dec_features = nmt_model.decoder(
            #     encoder_out=None,
            #     features_only=True,
            #     prev_output_tokens=trg_tokens,
            # )[0] # B x TT x dim(512)
            # net_input = {
            #     'src_tokens': torch.cat(
            #         (sample['net_input']['src_tokens'].unsqueeze(0), 
            #          sample['target'].unsqueeze(0)), 0
            #     ),
            #     'src_features': src_features,
            #     'trg_features': dec_features,
            #     'src_lengths': trg_lens,
            #     'prev_output_tokens': None,
            # }
            # new_sample = {
            #     'id': sample['id'],
            #     'nsentences': sample['nsentences'],
            #     'ntokens': trg_lens.sum().item(),
            #     'net_input': net_input,
            #     'target': trg_tokens.ne(pad_index)
            # }
            # return new_sample

        else:
            num_prev_tokens = 3  # How many previous tokens should we send in each time

            if unk_index in sample['action_seq']:
                # At least one sample is distorted
                new_action_seq = torch.empty(
                    sample['action_seq'].shape[0], sample['action_seq'].shape[1]
                ).fill_(pad_index)    # Use for prev_token
                target = torch.ones_like(sample['target'])  # Use for target
                for index, action_seq in enumerate(sample['action_seq']):
                    if unk_index not in action_seq:
                        new_action_seq[index] = action_seq
                        target[index] = action_seq[1:].contiguous()
                    else:
                        dist_index = (action_seq == unk_index).nonzero().squeeze()
                        new_action_seq[index][:-1] = action_seq[action_seq != unk_index]
                        target[index] = new_action_seq[index][1:].contiguous()
                        target[index][dist_index-2] = 9 - action_seq[dist_index-1]  # Turns 4 to 5 and vice versa
                padded_sample = torch.empty(
                    new_action_seq.shape[0], new_action_seq.shape[1] + num_prev_tokens - 1
                ).fill_(pad_index)
                prev_token = torch.empty(new_action_seq.shape[0], new_action_seq.shape[1] - 1, num_prev_tokens,
                                        device=sample['net_input']['src_tokens'].device)
                padded_sample[:, num_prev_tokens - 1:] = new_action_seq
            else:
                padded_sample = torch.empty(
                    sample['action_seq'].shape[0], sample['action_seq'].shape[1] + num_prev_tokens - 1
                ).fill_(pad_index)
                prev_token = torch.empty(sample['action_seq'].shape[0], sample['action_seq'].shape[1] - 1, num_prev_tokens,
                                        device=sample['net_input']['src_tokens'].device)
                target = sample['action_seq'][:, 1:].contiguous()
                padded_sample[:, num_prev_tokens-1:] = sample['action_seq']

            for i in range(num_prev_tokens):
                prev_token[:, :, i] = padded_sample[:, i:i-num_prev_tokens]
    else:
        pad_index = task.agent_dictionary.pad_index
        eos_index = task.agent_dictionary.eos_index
        unk_index = task.agent_dictionary.unk_index
        prev_token = sample['action_seq'][:, :-1]
        target = sample['action_seq'][:, 1:].contiguous()

    net_input = {
        'src_tokens': torch.cat((sample['net_input']['src_tokens'].unsqueeze(2),
                                 sample['target'].unsqueeze(2)), dim=2),
        'src_lengths': sample['net_input']['src_lengths'],
        'prev_output_tokens': prev_token.type(torch.LongTensor).to(device=sample['net_input']['src_tokens'].device)
    }
    new_sample = {
        'id': sample['id'],
        'nsentences': sample['nsentences'],
        'ntokens': sample['ntokens'],
        'net_input': net_input,
        'target': target
    }
    return new_sample


# no read token
# @torch.no_grad()
# def modify_preprocessed_input(sample, task, use_transformer_feature=False, nmt_model=None):
#     pad_index = task.agent_dictionary.pad_index
#     eos_index = task.agent_dictionary.eos_index
#     unk_index = task.agent_dictionary.unk_index

#     if task.cfg.arch.startswith("info"):

#         if use_transformer_feature:
#             bsz, seq_len = sample['action_seq'][:, 1:].size()
#             device = sample['target'].device
#             nmt_model.to(device)

#             cum_srcs = sample['net_input']['src_tokens'].clone()
#             cum_trgs = sample['target'].clone()

#             src_tokens = torch.full((bsz, seq_len), pad_index).to(device)
#             trg_tokens = torch.full((bsz, seq_len), pad_index).to(device)

#             read_nums = torch.full((bsz, seq_len), -1).to(device)
#             write_nums = torch.full((bsz, seq_len), -1).to(device)

#             for idx, act in enumerate(sample['action_seq']):
#                 act = act[act.eq(4) | act.eq(5)]
#                 act[act.eq(4)] = 1
#                 act[act.eq(5)] = 0
#                 act_len = act.size(-1) - 1
                
#                 raw_src = cum_srcs[idx, :act_len][act[:-1] == 1]
#                 src_tokens[idx, :raw_src.size(-1)] = raw_src

#                 raw_trg = cum_trgs[idx, :act_len][act[1:] == 0]
#                 trg_tokens[idx, :raw_trg.size(-1)] = raw_trg
                
#                 cum_read = act[:-1].cumsum(-1) - 1
#                 cum_read[cum_read < 0] = 0
#                 read_nums[idx, :cum_read.size(-1)] = cum_read

#                 cum_write = (~(act[1:].bool())).long().cumsum(-1) - 1
#                 cum_write[cum_write < 0] = 0
#                 write_nums[idx, :cum_write.size(-1)] = cum_write

#             mask = torch.arange(seq_len).unsqueeze(0).to(device)
#             mask = mask.repeat(seq_len * bsz, 1).view(bsz, seq_len, -1)
#             read_mask = read_nums.unsqueeze(-1).repeat(1, 1, seq_len)
#             write_mask = write_nums.unsqueeze(-1).repeat(1, 1, seq_len)

#             src_tokens = src_tokens.unsqueeze(1).repeat_interleave(seq_len, dim=1)
#             src_tokens.masked_fill_(mask > read_mask, pad_index)

#             trg_tokens = trg_tokens.unsqueeze(1).repeat_interleave(seq_len, dim=1)
#             trg_tokens.masked_fill_(mask > write_mask, pad_index)

#             read_token_idx = (trg_tokens != 1).sum(-1).unsqueeze(-1)
#             read_tokens = sample['target'].clone().unsqueeze(-1)
#             read_token_mask = (sample['action_seq'][:, 1:] == 4).unsqueeze(-1)
#             read_tokens[~read_token_mask] = 1
#             trg_tokens = torch.cat(
#                 (trg_tokens, torch.ones_like(trg_tokens[:, :, :1])), -1
#             )
#             trg_tokens.scatter_(-1, read_token_idx, read_tokens)
#             trg_tokens = trg_tokens[:, :, :-1]

#             src_tokens = src_tokens.view(-1, seq_len)
#             trg_tokens = trg_tokens.view(-1, seq_len)


#             # pad_mask = (sample['action_seq'][:, 1:].flatten() != 1)
#             valid_mask = src_tokens[:, 0].ne(pad_index)

#             if task.except_eos:
#                 eos_mask = src_tokens.eq(eos_index).any(-1) | trg_tokens.eq(eos_index).any(-1)
#                 valid_mask = valid_mask & (~eos_mask)

#             src_lenghts = (src_tokens != pad_index).sum(-1)
#             src_lenghts = src_lenghts[valid_mask]
#             trg_lenghts = (trg_tokens != pad_index).sum(-1)
#             trg_lenghts = trg_lenghts[valid_mask]

#             src_tokens = src_tokens[valid_mask]
#             enc_outs = nmt_model.encoder(
#                 src_tokens,
#                 src_lenghts,
#             )
#             src_features = enc_outs['encoder_out'][0] # L x B * L x dim(512)
#             src_features = src_features.transpose(1,0).contiguous()

#             trg_tokens = trg_tokens[valid_mask]
#             trg_features = nmt_model.decoder(
#                 encoder_out=None,
#                 features_only=True,
#                 prev_output_tokens=trg_tokens,
#             )[0] # B * L x L * dim(512)
            
#             assert torch.isnan(src_features).sum() == 0
#             assert torch.isnan(trg_features).sum() == 0

#             actions = sample['action_seq'][:, 1:].flatten()[valid_mask]
#             write_mask = actions.eq(5)
#             actions = actions[write_mask]
#             src_tokens = src_tokens[write_mask]
#             trg_tokens = trg_tokens[write_mask]
#             src_features = src_features[write_mask]
#             trg_features = trg_features[write_mask]
#             src_lenghts = src_lenghts[write_mask]
#             trg_lenghts = trg_lenghts[write_mask]

#             net_input = {
#                 'src_tokens': src_tokens,
#                 'trg_tokens': trg_tokens,
#                 'src_features': src_features,
#                 'trg_features': trg_features,
#                 'src_lengths': src_lenghts,
#                 'trg_lengths': trg_lenghts,
#                 'prev_output_tokens': None,
#             }
#             new_sample = {
#                 'id': sample['id'],
#                 'nsentences': (sample['nsentences'] * seq_len) - (~valid_mask).sum(),
#                 'ntokens': src_lenghts.sum(-1),
#                 'net_input': net_input,
#                 'target': actions,
#             }
#             return new_sample

#         else:
#             num_prev_tokens = 3  # How many previous tokens should we send in each time

#             if unk_index in sample['action_seq']:
#                 # At least one sample is distorted
#                 new_action_seq = torch.empty(
#                     sample['action_seq'].shape[0], sample['action_seq'].shape[1]
#                 ).fill_(pad_index)    # Use for prev_token
#                 target = torch.ones_like(sample['target'])  # Use for target
#                 for index, action_seq in enumerate(sample['action_seq']):
#                     if unk_index not in action_seq:
#                         new_action_seq[index] = action_seq
#                         target[index] = action_seq[1:].contiguous()
#                     else:
#                         dist_index = (action_seq == unk_index).nonzero().squeeze()
#                         new_action_seq[index][:-1] = action_seq[action_seq != unk_index]
#                         target[index] = new_action_seq[index][1:].contiguous()
#                         target[index][dist_index-2] = 9 - action_seq[dist_index-1]  # Turns 4 to 5 and vice versa
#                 padded_sample = torch.empty(
#                     new_action_seq.shape[0], new_action_seq.shape[1] + num_prev_tokens - 1
#                 ).fill_(pad_index)
#                 prev_token = torch.empty(new_action_seq.shape[0], new_action_seq.shape[1] - 1, num_prev_tokens,
#                                         device=sample['net_input']['src_tokens'].device)
#                 padded_sample[:, num_prev_tokens - 1:] = new_action_seq
#             else:
#                 padded_sample = torch.empty(
#                     sample['action_seq'].shape[0], sample['action_seq'].shape[1] + num_prev_tokens - 1
#                 ).fill_(pad_index)
#                 prev_token = torch.empty(sample['action_seq'].shape[0], sample['action_seq'].shape[1] - 1, num_prev_tokens,
#                                         device=sample['net_input']['src_tokens'].device)
#                 target = sample['action_seq'][:, 1:].contiguous()
#                 padded_sample[:, num_prev_tokens-1:] = sample['action_seq']

#             for i in range(num_prev_tokens):
#                 prev_token[:, :, i] = padded_sample[:, i:i-num_prev_tokens]
#     else:
#         prev_token = sample['action_seq'][:, :-1]
#         target = sample['action_seq'][:, 1:].contiguous()

#     net_input = {
#         'src_tokens': torch.cat((sample['net_input']['src_tokens'].unsqueeze(2),
#                                  sample['target'].unsqueeze(2)), dim=2),
#         'src_lengths': sample['net_input']['src_lengths'],
#         'prev_output_tokens': prev_token.type(torch.LongTensor).to(device=sample['net_input']['src_tokens'].device)
#     }
#     new_sample = {
#         'id': sample['id'],
#         'nsentences': sample['nsentences'],
#         'ntokens': sample['ntokens'],
#         'net_input': net_input,
#         'target': target
#     }
#     return new_sample
