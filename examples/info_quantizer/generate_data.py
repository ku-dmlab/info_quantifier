#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import os
import sys
from pathlib import Path
from itertools import chain
from argparse import Namespace
from omegaconf import DictConfig

import sacrebleu
import torch
import numpy as np

from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.data import encoders
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter

from examples.info_quantizer.data import agent_utils
from examples.info_quantizer.modules.latency import length_adaptive_average_lagging, AverageLagging

def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
            not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
            cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        target = 'target' if cfg.task.has_target else 'no_target'
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}-{}-beam{}.txt".format(cfg.dataset.gen_subset, target, cfg.generation.beam),
        )
        
        score_path = Path(os.path.join(
            cfg.common_eval.results_path,
            "score-{}-{}-beam{}.txt".format(cfg.dataset.gen_subset, target, cfg.generation.beam),
        ))
        if score_path.exists():
            os.remove(score_path)
        open(score_path, 'w').close()

        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h, score_path)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}

def detok(sents, dict, post_process):
    return [
        dict.string(
            sent,
            bpe_symbol=post_process,
            extra_symbols_to_ignore={dict.pad(), dict.eos()}
        )
        for sent in sents
    ]
def hf_detok(sents, hf_model):
    tokenizer = hf_model.tokenizer
    return tokenizer.batch_decode(sents, skip_special_tokens=True)

def _main(cfg: DictConfig, output_file, score_path=None):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))

    hf_model = task.cfg.hf_model
    if hf_model:
        models = [task.nmt_model]
        task.load_dataset(cfg.dataset.gen_subset)
    else:
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )
        # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
        task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if not hf_model:
            if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(cfg)
            model.decoder.waitk = task.eval_waitk
        else:
            model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=(200, 200),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
    seq_gen_cls = None
    if hf_model:
        from examples.info_quantizer.modules.action_generator_hf import HfActionGenerator
        seq_gen_cls = HfActionGenerator
    generator = task.build_action_generator(
        models, cfg.generation, seq_gen_cls=seq_gen_cls, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(cfg.tokenizer)
    bpe = encoders.build_bpe(cfg.bpe)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    scorer = scoring.build_scorer(cfg.scoring, tgt_dict)

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()

    als, laals = [], []
    total_hyps, total_trgs = [], []

    for sample in progress:
        sample_ids = sample['id'].clone()
        detok_trg_strs = [
            tgt_dict.string(
                tgt_,
                bpe_symbol=cfg.common_eval.post_process,
                extra_symbols_to_ignore={tgt_dict.pad(), tgt_dict.eos()}
            )
            for tgt_ in sample['target']
        ]
        target_lens = [tgt_[tgt_.ne(tgt_dict.pad())].size(-1) for tgt_ in sample['target']]


        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()
        task.cfg.arch = "agent_lstm_0"
        hypos = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )

        def extend_with_src_trg_tokens(inputs, hyps):
            inputs['input_src'] = inputs['net_input']['src_tokens'][:, :, 0]
            inputs['input_trg'] = inputs['net_input']['src_tokens'][:, :, 1]
            new_target = torch.empty(inputs['target'].shape[0], inputs['target'].shape[1] + 1,
                                     dtype=torch.long).fill_(4)
            new_target[:, 1:] = inputs['target']
            inputs['input_act'] = new_target
            return inputs

        if not hypos:
            continue
        
        # generated_input, iq_input = agent_utils.prepare_simultaneous_input(
        #     hypos, sample, task, task.nmt_model.get_trg_features
        # )
        generated_input, iq_input = agent_utils.prepare_simultaneous_input(hypos, sample, task)
        generated_input = extend_with_src_trg_tokens(generated_input, hypos)

        num_generated_tokens = sum(len(h['tokens']) for h in hypos)
        gen_timer.stop(num_generated_tokens)
        
        for i, input_id in enumerate(generated_input["id"]):
            has_target = generated_input["input_act"] is not None

            action_tokens = None
            if has_target:
                action_tokens = (
                    utils.strip_pad(generated_input["input_act"][i, :], task.agent_dictionary.pad()).int().cpu()
                )

            action_len = len(action_tokens)
            src_tokens = generated_input['input_src'][i, :action_len-1]
            trg_tokens = generated_input['input_trg'][i, :action_len-1]

            src_str = src_dict.string(src_tokens, include_eos=True)
            trg_str = tgt_dict.string(
                trg_tokens,
                include_eos=True,
                escape_unk=False,
            )
            action_str = " ".join([str(act.item()) for act in action_tokens])

            #assert len(src_str.split()) == len(trg_str.split()) and \
            #       len(action_str.split())-1 == len(trg_str.split())

            if src_dict is not None:
                print("S-{}\t{}".format(input_id, src_str), file=output_file)
            if has_target:
                print("T-{}\t{}".format(input_id, trg_str), file=output_file)

            # Generated action sequence
            print(
                "A-{}\t{}".format(input_id, action_str),
                file=output_file,
            )

            ### Score Check
            if cfg.task.distort_action_sequence:
                action_tokens = action_tokens[action_tokens != 3]
                action_len = action_tokens.size(-1)
                src_tokens = generated_input['input_src'][i, :action_len-1]
                trg_tokens = generated_input['input_trg'][i, :action_len-1]

            write_tokens = trg_tokens[action_tokens[1:].eq(5)]
            detok_hyp_str = tgt_dict.string(
                write_tokens,
                bpe_symbol=cfg.common_eval.post_process,
                extra_symbols_to_ignore={tgt_dict.pad(), tgt_dict.eos()}
            )
            total_hyps.append(detok_hyp_str)

            action_seq = action_tokens.long()[1:].clone()
            action_seq = action_seq[action_seq.eq(4) | action_seq.eq(5)]
            action_seq[action_seq.eq(5)] = 0 # write
            action_seq[action_seq.eq(4)] = 1 # read

            delay = action_seq.cumsum(-1) + 1
            delay = delay[action_seq.eq(0)]
            assert delay.size(-1) == write_tokens.size(-1)
            cur_idx = sample_ids.tolist().index(input_id)
            ref_len = target_lens[cur_idx]
            total_trgs.append(detok_trg_strs[cur_idx])
            laal = length_adaptive_average_lagging(
                delays=delay.unsqueeze(0), 
                src_lens=torch.tensor([[delay.max()]]),
                tgt_lens=torch.tensor([[delay.size(-1)]]),
                ref_lens=torch.tensor([[ref_len]]),
            ).item()
            al = AverageLagging(
                delay.unsqueeze(0), 
                torch.tensor([[delay.max()]]),
                torch.tensor([[delay.size(-1)]]),
                torch.tensor([[ref_len]]),
            ).item()
            laals.append(laal)
            als.append(al)
            sent_bleu = sacrebleu.sentence_bleu(detok_hyp_str, [detok_trg_strs[cur_idx]]).score

            with open(score_path, 'a') as sf:
                sf.write(f"ID:{input_id}:B:{sent_bleu}:AL:{al}:LAAL:{laal}\n")

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += (
            sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
        )
        
        del hypos
        del generated_input
        del iq_input
        torch.cuda.empty_cache()


    corpus_bleu = sacrebleu.corpus_bleu(total_hyps, [total_trgs])
    al_mean = np.mean(als)
    laal_mean = np.mean(laals)
    with open(score_path, 'a') as sf:
        sf.write(f"{corpus_bleu}\n")
        sf.write(f"AL:{al_mean}:LAAL:{laal_mean}")

        
    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info(
        "Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
            )
    )
    print(
        "Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
            ), file=output_file
    )
    if has_target:
        if cfg.bpe and not cfg.generation.sacrebleu:
            if cfg.common_eval.post_process:
                logger.warning(
                    "BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization"
                )
            else:
                logger.warning(
                    "If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization"
                )

    return scorer


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()