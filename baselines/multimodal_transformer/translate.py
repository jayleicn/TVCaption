""" Translate input text with trained model. """

import os
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import random
import numpy as np
import subprocess

from baselines.multimodal_transformer.translator import Translator
from baselines.multimodal_transformer.transformer.tvc_dataset import \
    caption_collate, prepare_batch_inputs
from baselines.multimodal_transformer.transformer.tvc_dataset import TVCaptionDataset
from utils.basic_utils import save_json, save_jsonl


def sort_res(res_dict):
    """res_dict: the submission json entry `results`"""
    final_res_dict = {}
    for k, v in res_dict.items():
        final_res_dict[k] = sorted(v, key=lambda x: float(x["timestamp"][0]))
    return final_res_dict


def run_translate(eval_data_loader, translator, opt):
    # submission template
    batch_res = []
    with torch.no_grad():
        for batch in tqdm(eval_data_loader, mininterval=2, desc="  - (Translate)"):
            model_inputs = prepare_batch_inputs(batch[0], device=opt.device, non_blocking=opt.pin_memory)
            meta = batch[1]

            dec_seq = translator.translate_batch(model_inputs,
                                                 max_cap_len=opt.max_cap_len,
                                                 use_beam=opt.use_beam)

            # example_idx indicates which example is in the batch
            for example_idx, (cur_gen_sen, cur_meta) in enumerate(zip(dec_seq, meta)):
                cur_data = {
                    "descs": [
                        {"desc": eval_data_loader.dataset.convert_ids_to_sentence(
                         cur_gen_sen.cpu().tolist()).encode("ascii", "ignore")}
                              ],
                    "clip_id": cur_meta["clip_id"],
                    "ts": cur_meta["ts"],
                    "vid_name": cur_meta["vid_name"]
                }
                batch_res.append(cur_data)

            if opt.debug:
                break
    return batch_res


def get_data_loader(opt):
    eval_dataset = TVCaptionDataset(
        ctx_mode=opt.ctx_mode,
        data_path=opt.eval_path,
        sub_meta_path=opt.sub_meta_path,
        vid_h5_path_or_handler=opt.vid_feat_path,
        word2idx_path=opt.word2idx_path,
        max_cap_len=opt.max_cap_len,
        max_sub_len=opt.max_sub_len,
        max_v_len=opt.max_v_len,
        h5driver=opt.h5driver,
        clip_length=1.5,
        normalize_vfeat=not opt.no_norm_vfeat,
        is_eval=True
    )
    eval_data_loader = DataLoader(eval_dataset,
                                  collate_fn=caption_collate,
                                  batch_size=opt.batch_size,
                                  shuffle=False,
                                  num_workers=opt.num_workers,
                                  pin_memory=opt.pin_memory)
    return eval_data_loader


def main():
    parser = argparse.ArgumentParser(description="translate.py")

    parser.add_argument("-eval_split_name", choices=["val", "test_public"])
    parser.add_argument("-eval_path", type=str, help="Path to eval data")
    parser.add_argument("-reference_path", type=str, default=None, help="Path to reference")
    parser.add_argument("-res_dir", required=True, help="path to dir containing model .pt file")
    parser.add_argument("-batch_size", type=int, default=100, help="batch size")

    # beam search configs
    parser.add_argument("-use_beam", action="store_true", help="use beam search, otherwise greedy search")
    parser.add_argument("-beam_size", type=int, default=2, help="beam size")
    parser.add_argument("-n_best", type=int, default=1, help="stop searching when get n_best from beam search")
    parser.add_argument("-min_sen_len", type=int, default=8, help="minimum length of the decoded sentences")
    parser.add_argument("-max_sen_len", type=int, default=25, help="maximum length of the decoded sentences")
    parser.add_argument("-block_ngram_repeat", type=int, default=0, help="block repetition of ngrams during decoding.")
    parser.add_argument("-length_penalty_name", default="none",
                        choices=["none", "wu", "avg"], help="length penalty to use.")
    parser.add_argument("-length_penalty_alpha", type=float, default=0.,
                        help="Google NMT length penalty parameter (higher = longer generation)")

    parser.add_argument("-no_cuda", action="store_true")
    parser.add_argument("-seed", default=2019, type=int)
    parser.add_argument("-debug", action="store_true")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    checkpoint = torch.load(os.path.join(opt.res_dir, "model.chkpt"))

    decoding_strategy = "beam{}_lp_{}_la_{}".format(
        opt.beam_size, opt.length_penalty_name, opt.length_penalty_alpha) if opt.use_beam else "greedy"
    save_json(vars(opt), os.path.join(opt.res_dir, "{}_eval_cfg.json".format(decoding_strategy)), save_pretty=True)

    # add some of the train configs
    train_opt = checkpoint["opt"]  # EDict(load_json(os.path.join(opt.res_dir, "model.cfg.json")))
    for k in train_opt.__dict__:
        if k not in opt.__dict__:
            setattr(opt, k, getattr(train_opt, k))

    if "ctx_mode" not in opt:
        opt.ctx_mode = "video_sub"  # temp hack, since the first experiment does not have such a setting

    eval_data_loader = get_data_loader(opt)

    # setup model
    translator = Translator(opt, checkpoint)

    pred_file = os.path.join(opt.res_dir, "{}_pred_{}.jsonl".format(decoding_strategy, opt.eval_split_name))
    pred_file = os.path.abspath(pred_file)
    if not os.path.exists(pred_file):
        json_res = run_translate(eval_data_loader, translator, opt=opt)
        save_jsonl(json_res, pred_file)
    else:
        print("Using existing prediction file at {}".format(pred_file))

    if opt.reference_path:
        # COCO language evaluation
        reference_path = os.path.abspath(opt.reference_path)
        metrics_path = pred_file.replace(".json", "_lang_metrics.json")
        eval_cmd = ["python", "evaluate.py", "-s", pred_file, "-o", metrics_path,
                    "-r", reference_path]
        subprocess.call(eval_cmd, cwd="standalone_eval")

    print("[Info] Finished {}.".format(opt.eval_split_name))


if __name__ == "__main__":
    main()
