import os
from fairseq import checkpoint_utils, tasks


def load_mt_model(model_path, return_dict=False):
    if not os.path.exists(model_path):
        raise IOError("Model file not found: {}".format(model_path))

    state = checkpoint_utils.load_checkpoint_to_cpu(model_path)
    task_args = state["cfg"]["task"]
    task = tasks.setup_task(task_args)

    # build model for ensemble
    state["cfg"]["model"].load_pretrained_encoder_from = None
    state["cfg"]["model"].load_pretrained_decoder_from = None

    mt_model = task.build_model(state["cfg"]["model"])
    mt_model.load_state_dict(state["model"], strict=True)
    mt_model.eval()
    mt_model.share_memory()
    # mt_model.cuda()

    if return_dict:
        return mt_model, task.target_dictionary, task.source_dictionary
    else:
        return mt_model
