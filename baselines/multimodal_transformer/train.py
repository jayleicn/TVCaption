"""
This script handling the training process.
"""

import argparse
import math
import time

import random
import numpy as np
import os
import subprocess
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from baselines.multimodal_transformer.transformer.tvc_dataset import \
    caption_collate, prepare_batch_inputs
from baselines.multimodal_transformer.transformer.tvc_dataset import TVCaptionDataset
from baselines.multimodal_transformer.transformer.model import MMT
from baselines.multimodal_transformer.transformer.optimization import BertAdam
from baselines.multimodal_transformer.translator import Translator
from baselines.multimodal_transformer.translate import run_translate
from utils.basic_utils import save_json, load_json, save_jsonl, count_parameters
from easydict import EasyDict as EDict
from tensorboardX import SummaryWriter
import pprint
import logging
logger = logging.getLogger(__name__)


def cal_performance(pred, gold):
    pred = pred.max(2)[1].contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    valid_label_mask = gold.ne(TVCaptionDataset.IGNORE)
    pred_correct_mask = pred.eq(gold)
    n_correct = pred_correct_mask.masked_select(valid_label_mask).sum().item()
    return n_correct


def train_epoch(model, training_data, optimizer, opt, epoch):
    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    torch.autograd.set_detect_anomaly(True)
    for batch_idx, batch in tqdm(enumerate(training_data), mininterval=2,
                                 desc="  Training =>", total=len(training_data)):
        niter = epoch * len(training_data) + batch_idx
        # prepare data
        model_inputs = prepare_batch_inputs(batch[0], device=opt.device, non_blocking=opt.pin_memory)
        # forward & backward
        optimizer.zero_grad()
        loss, pred_scores = model(**model_inputs)

        # make it consistent with other configs
        pred_scores_list = [pred_scores]
        input_labels_list = [model_inputs["caption_labels"]]

        loss.backward()
        if opt.grad_clip != -1:  # enable, -1 == disable
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        # keep logs
        n_correct = 0
        n_word = 0
        for pred, gold in zip(pred_scores_list, input_labels_list):
            n_correct += cal_performance(pred, gold)
            valid_label_mask = gold.ne(TVCaptionDataset.IGNORE)
            n_word += valid_label_mask.sum().item()

        n_word_total += n_word
        n_word_correct += n_correct
        cur_loss = loss.item()
        total_loss += cur_loss

        if opt.debug:
            break
    torch.autograd.set_detect_anomaly(False)

    loss_per_word = 1.0 * total_loss / n_word_total
    accuracy = 1.0 * n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval_language_metrics(checkpoint, eval_data_loader, opt, model=None, eval_mode="val"):
    """eval_mode can only be set to `val` here, as setting to `test` is cheating
    0, run inference
    1, Get METEOR, BLEU1-4, CIDEr scores
    2, Get vocab size, sentence length
    """
    translator = Translator(opt, checkpoint, model=model)
    json_res = run_translate(eval_data_loader, translator, opt=opt)
    res_filepath = os.path.abspath(opt.save_model + "_tmp_greedy_pred_{}.json".format(eval_mode))
    save_jsonl(json_res, res_filepath)

    # COCO language evaluation
    reference_path = os.path.abspath(opt.reference_path)
    metrics_path = res_filepath.replace(".json", "_lang_metrics.json")
    eval_cmd = ["python", "evaluate.py", "-s", res_filepath, "-o", metrics_path,
                "-r", reference_path]
    subprocess.call(eval_cmd, cwd="standalone_eval")
    metrics = load_json(metrics_path)
    return metrics, [res_filepath, metrics_path]


def train(model, training_data, validation_data, opt):
    model = model.to(opt.device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    num_train_optimization_steps = len(training_data) * opt.n_epoch
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=opt.lr,
                         warmup=opt.lr_warmup_proportion,
                         t_total=num_train_optimization_steps,
                         schedule="warmup_linear")

    writer = SummaryWriter(opt.res_dir)
    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + ".train.log"
        log_valid_file = opt.log + ".valid.log"

        logger.info("Training performance will be written to file: {} and {}".format(
            log_train_file, log_valid_file))

        with open(log_train_file, "w") as log_tf, open(log_valid_file, "w") as log_vf:
            log_tf.write("epoch,loss,ppl,accuracy\n")
            log_vf.write("epoch,loss,ppl,accuracy,METEOR,BLEU@4,CIDEr,re4\n")

    prev_best_score = 0.
    es_cnt = 0
    for epoch_i in range(opt.n_epoch):
        logger.info("[Epoch {}]".format(epoch_i))

        start = time.time()
        train_loss, train_acc = train_epoch(model, training_data, optimizer, opt, epoch_i)
        logger.info("[Training]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %, elapse {elapse:3.3f} min"
                    .format(ppl=math.exp(min(train_loss, 100)), acc=100*train_acc, elapse=(time.time()-start)/60.))
        niter = (epoch_i + 1) * len(training_data)  # number of bart
        writer.add_scalar("Train/Acc", train_acc, niter)
        writer.add_scalar("Train/Loss", train_loss, niter)

        # Note here we use greedy generated words to predicted next words, the true inference situation.
        checkpoint = {
            "model": model.state_dict(),
            "model_cfg": model.config,
            "opt": opt,
            "epoch": epoch_i}

        val_greedy_output, filepaths = eval_language_metrics(
            checkpoint, validation_data, opt, eval_mode="val", model=model)
        cider = val_greedy_output["CIDEr"]
        logger.info("[Val] METEOR {m:.2f} Bleu@4 {b:.2f} CIDEr {c:.2f}"
                    .format(m=val_greedy_output["METEOR"]*100,
                            b=val_greedy_output["Bleu_4"]*100,
                            c=val_greedy_output["CIDEr"]*100))
        writer.add_scalar("Val/METEOR", val_greedy_output["METEOR"]*100, niter)
        writer.add_scalar("Val/Bleu_4", val_greedy_output["Bleu_4"]*100, niter)
        writer.add_scalar("Val/CIDEr", val_greedy_output["CIDEr"]*100, niter)

        if opt.save_mode == "all":
            model_name = opt.save_model + "_acc_{c}.chkpt".format(c=cider*100)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == "best":
            model_name = opt.save_model + ".chkpt"
            if cider > prev_best_score:
                es_cnt = 0
                prev_best_score = cider
                torch.save(checkpoint, model_name)
                new_filepaths = [e.replace("tmp", "best") for e in filepaths]
                for src, tgt in zip(filepaths, new_filepaths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if es_cnt > opt.max_es_cnt:  # early stop
                    logger.info("Early stop at {} with CIDEr {}".format(epoch_i, prev_best_score))
                    break

        if log_train_file and log_valid_file:
            with open(log_train_file, "a") as log_tf, open(log_valid_file, "a") as log_vf:
                log_tf.write("{epoch},{loss: 8.5f},{ppl: 8.5f},{acc:3.3f}\n".format(
                    epoch=epoch_i, loss=train_loss, ppl=math.exp(min(train_loss, 100)), acc=100*train_acc))
                log_vf.write("{epoch},{m:.2f},{b:.2f},{c:.2f}\n".format(
                    epoch=epoch_i,
                    m=val_greedy_output["METEOR"]*100,
                    b=val_greedy_output["Bleu_4"]*100,
                    c=val_greedy_output["CIDEr"]*100))

        if opt.debug:
            break

    writer.close()


def get_args():
    """parse and preprocess cmd line args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-ctx_mode", type=str, default="video_sub", choices=["video", "sub", "video_sub"])

    # model config
    parser.add_argument("-hidden_size", type=int, default=768)
    parser.add_argument("-intermediate_size", type=int, default=768)
    parser.add_argument("-word_vec_size", type=int, default=300)
    parser.add_argument("-vid_feat_size", type=int, default=3072, help="2048 appearance + 1024 flow")
    parser.add_argument("-max_v_len", type=int, default=20, help="max length of video feature")
    parser.add_argument("-max_sub_len", type=int, default=50, help="max number of words in subtitle")
    parser.add_argument("-max_cap_len", type=int, default=20, help="max length of caption")
    parser.add_argument("-type_vocab_size", type=int, default=2, help="video as 0, text as 1")
    parser.add_argument("-layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("-hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("-num_hidden_layers", type=int, default=2, help="number of transformer layers")
    parser.add_argument("-attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("-num_attention_heads", type=int, default=12)
    parser.add_argument("-initializer_range", type=float, default=0.02)
    parser.add_argument("-glove_path", type=str, default=None, help="extracted GloVe vectors")
    parser.add_argument("-freeze_glove", action="store_true", help="do not train GloVe vectors")
    parser.add_argument("-share_wd_cls_weight", action="store_true",
                        help="share weight matrix of the word embedding with the final classifier, ")

    # training config -- learning rate
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-lr_warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument("-grad_clip", type=float, default=1, help="clip gradient, -1 == disable")

    parser.add_argument("-train_path", type=str, default=None, help="path to the training data")
    parser.add_argument("-eval_path", type=str, default=None, help="path to the eval data")
    parser.add_argument("-data_ratio", type=float, default=1, help="how many train/eval data to use")
    parser.add_argument("-reference_path", type=str,)
    parser.add_argument("-sub_meta_path", type=str, default=None, help="path to")
    parser.add_argument("-vid_feat_path", type=str, default=None, help="path to video features")
    parser.add_argument("-no_norm_vfeat", action="store_true",
                        help="Do not do normalization on video feat, use it when using i3d_resnet concat feat")
    parser.add_argument("-word2idx_path", type=str, default="./cache/word2idx.json")
    parser.add_argument("-label_smoothing", type=float, default=0.1,
                        help="Use soft target instead of one-hot hard target")
    parser.add_argument("-n_epoch", type=int, default=50, help="Number of training epochs")
    parser.add_argument("-max_es_cnt", type=int, default=10,
                        help="stop if the model is not improving for max_es_cnt max_es_cnt")
    parser.add_argument("-batch_size", type=int, default=128, help="training batch size")
    parser.add_argument("-eval_batch_size", type=int, default=50, help="inference batch size")

    parser.add_argument("-use_beam", action="store_true", help="use beam search, otherwise greedy search")
    parser.add_argument("-beam_size", type=int, default=2, help="beam size")
    parser.add_argument("-n_best", type=int, default=1, help="stop searching when get n_best from beam search")

    # others
    parser.add_argument("-exp_id", type=str, default="res", help="id of the current run")
    parser.add_argument("-res_root_dir", type=str, default="results", help="dir to containing all the results")
    parser.add_argument("-save_model", default="model")
    parser.add_argument("-save_mode", type=str, choices=["all", "best"], default="best",
                        help="all: save models at each epoch; best: only save the best model")
    parser.add_argument("-device", type=int, default=0, help="0 cuda, -1 cpu")
    parser.add_argument("-num_workers", type=int, default=8,
                         help="num subprocesses used to load the data, 0: use main process")
    parser.add_argument("-no_core_driver", action="store_true",
                        help="hdf5 driver, default use `core` (load into RAM), if specified, use `None`")
    parser.add_argument("-no_pin_memory", action="store_true",
                         help="Don't use pin_memory=True for dataloader. "
                              "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")
    parser.add_argument("-seed", default=2019, type=int)
    parser.add_argument("-debug", action="store_true")

    opt = parser.parse_args()
    # make paths
    if opt.debug:
        opt.res_root_dir = os.path.sep.join(opt.res_root_dir.split(os.path.sep)[:-1] + ["debug_results", ])

    opt.res_dir = os.path.join(opt.res_root_dir,
                               "-".join([opt.ctx_mode, opt.exp_id, time.strftime("%Y_%m_%d_%H_%M_%S")]))

    if os.path.exists(opt.res_dir):
        raise ValueError("File exists {}".format(opt.res_dir))
    else:
        os.makedirs(opt.res_dir)

    opt.log = os.path.join(opt.res_dir, opt.save_model)
    opt.save_model = os.path.join(opt.res_dir, opt.save_model)

    if opt.share_wd_cls_weight:
        assert opt.word_vec_size == opt.hidden_size, \
            "hidden size has to be the same as word embedding size when " \
            "sharing the word embedding weight and the final classifier weight"

    cfg_name = opt.save_model + ".cfg.json"
    args_dict = vars(opt)
    save_json(args_dict, cfg_name, save_pretty=True)

    opt.h5driver = None if opt.no_core_driver else "core"
    opt.num_workers = 1 if opt.no_core_driver else opt.num_workers
    opt.pin_memory = not opt.no_pin_memory
    opt.device = torch.device("cuda:0" if opt.device >= 0 else "cpu")

    if opt.vid_feat_size > 3000:  # 3072, the normalized concatenation of resnet+i3d
        assert opt.no_norm_vfeat
    return opt


def main():
    opt = get_args()

    # random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    train_dataset = TVCaptionDataset(
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        data_path=opt.train_path,
        sub_meta_path=opt.sub_meta_path,
        vid_h5_path_or_handler=opt.vid_feat_path,
        word2idx_path=opt.word2idx_path,
        max_cap_len=opt.max_cap_len,
        max_sub_len=opt.max_sub_len,
        max_v_len=opt.max_v_len,
        h5driver=opt.h5driver,
        clip_length=1.5,
        normalize_vfeat=not opt.no_norm_vfeat,
        is_eval=False
    )
    eval_dataset = TVCaptionDataset(
        ctx_mode=opt.ctx_mode,
        # data_ratio=opt.data_ratio,
        data_ratio=1.0,
        data_path=opt.eval_path,
        sub_meta_path=opt.sub_meta_path,
        vid_h5_path_or_handler=train_dataset.vid_h5 if "video" in opt.ctx_mode else None,
        word2idx_path=opt.word2idx_path,
        max_cap_len=opt.max_cap_len,
        max_sub_len=opt.max_sub_len,
        max_v_len=opt.max_v_len,
        h5driver=opt.h5driver,
        clip_length=1.5,
        normalize_vfeat=not opt.no_norm_vfeat,
        is_eval=True  # only set to True at inference
    )

    train_loader = DataLoader(train_dataset,
                              collate_fn=caption_collate,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers,
                              pin_memory=opt.pin_memory)
    eval_loader = DataLoader(eval_dataset,
                             collate_fn=caption_collate,
                             batch_size=opt.eval_batch_size,
                             shuffle=False,
                             num_workers=opt.num_workers,
                             pin_memory=opt.pin_memory)

    opt.vocab_size = len(train_dataset.word2idx)
    pprint.pprint(vars(opt))

    rt_config = EDict(
        hidden_size=opt.hidden_size,
        intermediate_size=opt.intermediate_size,  # after each self attention
        vocab_size=opt.vocab_size,  # get from word2idx
        word_vec_size=opt.word_vec_size,
        video_feature_size=opt.vid_feat_size,
        max_position_embeddings=max(opt.max_v_len + opt.max_sub_len, opt.max_cap_len),  # get from max_seq_len
        type_vocab_size=opt.type_vocab_size,
        layer_norm_eps=opt.layer_norm_eps,  # bert layernorm
        hidden_dropout_prob=opt.hidden_dropout_prob,  # applies everywhere except attention
        num_hidden_layers=opt.num_hidden_layers,  # number of transformer layers
        num_attention_heads=opt.num_attention_heads,
        attention_probs_dropout_prob=opt.attention_probs_dropout_prob,  # applies only to self attention
        initializer_range=opt.initializer_range,
        label_smoothing=opt.label_smoothing,
        share_wd_cls_weight=opt.share_wd_cls_weight
    )
    model = MMT(rt_config)

    if opt.glove_path is not None:
        if hasattr(model, "embeddings"):
            logger.info("Load GloVe as word embedding")
            model.embeddings.set_pretrained_embedding(
                torch.from_numpy(torch.load(opt.glove_path)).float(), freeze=opt.freeze_glove)
        else:
            logger.warning("This model has no embeddings, cannot load glove vectors into the model")

    train(model, train_loader, eval_loader, opt)


if __name__ == "__main__":
    main()
