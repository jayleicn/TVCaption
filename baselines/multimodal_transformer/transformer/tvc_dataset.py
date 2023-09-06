import torch
import logging
import math
import nltk
import numpy as np

import h5py
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from utils.basic_utils import load_json, load_jsonl, flat_list_of_lists, l2_normalize_np_array

log_format = "%(asctime)-10s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


class TVCaptionDataset(Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
    SEP_TOKEN = "[SEP]"  # a separator for video and text
    VID_TOKEN = "[VID]"  # used as placeholder in the clip+text joint sequence
    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    UNK_TOKEN = "[UNK]"
    PAD = 0
    CLS = 1
    SEP = 2
    VID = 3
    BOS = 4
    EOS = 5
    UNK = 6
    IGNORE = -1  # used to calculate loss

    def __init__(self, ctx_mode, data_path, sub_meta_path, vid_h5_path_or_handler, word2idx_path,
                 max_cap_len, max_v_len, max_sub_len, h5driver=None, clip_length=1.5,
                 normalize_vfeat=True, is_eval=False, data_ratio=1.0):

        self.ctx_mode = ctx_mode
        self.use_video = "video" in ctx_mode
        self.use_sub = "sub" in ctx_mode
        self.is_eval = is_eval
        self.data_ratio = data_ratio
        self.word2idx = load_json(word2idx_path)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.clip_length = clip_length
        self.sub_meta_path = sub_meta_path
        self.max_v_len = max_v_len
        self.max_cap_len = max_cap_len  # sen
        self.max_sub_len = max_sub_len
        self.normalize_vfeat = normalize_vfeat

        if self.use_video:
            if isinstance(vid_h5_path_or_handler, h5py.File):
                self.vid_h5 = vid_h5_path_or_handler
            else:
                self.vid_h5 = h5py.File(vid_h5_path_or_handler, "r", driver=h5driver)

        if self.use_sub:
            self.sub_meta_dict = load_process_sub_meta(sub_meta_path, clip_length)

        self.word2idx = load_json(word2idx_path)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.data = self._load_data(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, meta = self.convert_example_to_features(self.data[index])
        return data, meta

    def convert_example_to_features(self, example):
        """
        {"vid_name": str,
         "duration": float,
         "ts": [st(float), ed(float)],
         "desc": str,
         "clip_id": str
        }
        """
        ctx_input_ids, ctx_input_mask, ctx_token_type_ids, video_feature, video_mask = self.get_context(example)
        if self.is_eval:
            caption_input_ids = [0, 0]
            caption_mask = [0, 0]
            caption_labels = [-1, -1]
        else:
            caption_input_ids, caption_mask = self.get_caption(example)
            # shifted right, `-1` is ignored when calculating CrossEntropy Loss
            caption_labels = [self.IGNORE if m == 0 else tid
                              for tid, m in zip(caption_input_ids, caption_mask)][1:] + [self.IGNORE]

        data = dict(
            caption_input_ids=np.array(caption_input_ids).astype(np.int64),
            caption_mask=np.array(caption_mask).astype(np.float32),
            caption_labels=np.array(caption_labels).astype(np.int64),
            ctx_input_ids=np.array(ctx_input_ids).astype(np.int64),
            ctx_input_mask=np.array(ctx_input_mask).astype(np.float32),
            ctx_token_type_ids=np.array(ctx_token_type_ids).astype(np.int64),
            video_feature=video_feature.astype(np.float32),
        )
        meta = example
        return data, meta

    def _load_data(self, data_path):
        logging.info("Loading data from {}".format(data_path))
        raw_lines = load_jsonl(data_path)
        raw_lines = raw_lines[:int(len(raw_lines) * self.data_ratio)]
        data = []
        for line in raw_lines:
            if self.is_eval:
                data.append(dict(
                    vid_name=line["vid_name"],
                    duration=line["duration"],
                    ts=line["ts"],
                    clip_id=line["clip_id"],
                    clip_st_ed=self.convert_ts_to_clip_indices(line["ts"])
                ))
            else:
                for d in line["descs"]:
                    data.append(dict(
                        vid_name=line["vid_name"],
                        duration=line["duration"],
                        ts=line["ts"],
                        clip_id=line["clip_id"],
                        desc=d["desc"],
                        clip_st_ed=self.convert_ts_to_clip_indices(line["ts"])
                    ))

        logging.info("Loading complete! {} captions".format(len(data)))
        return data

    def _tokenize_and_pad_sentence(self, sentence, max_sen_l):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_cap_len
        All non-PAD values are valid, with a mask value of 1
        """
        sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_sen_l - 2]
        sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_sen_l - valid_l)
        sentence_tokens += [self.PAD_TOKEN] * (max_sen_l - valid_l)
        return sentence_tokens, mask

    def _load_indexed_sub(self, sub_meta, clip_st_ed):
        st, ed = clip_st_ed
        sub_sentence_indices = sorted(list(set(flat_list_of_lists(
            [sub_meta["clip2sen"][str(idx)] for idx in range(st, ed+1, 1)
             if str(idx) in sub_meta["clip2sen"]]))))  # increasing order

        sub_sentences = " ".join([sub_meta["sub"][idx]["text"] for idx in sub_sentence_indices])
        sub_tokens, sub_mask = self._tokenize_and_pad_sentence(sub_sentences, max_sen_l=self.max_sub_len)
        return sub_tokens, sub_mask

    def _load_indexed_video_feature(self, raw_feat, clip_st_ed):
        """ [CLS], [VID], ..., [VID], [SEP], [PAD], ..., [PAD],
        All non-PAD tokens are valid, will have a mask value of 1.
        Returns:
            feat is padded to length of (self.max_v_len + self.max_cap_len,)
            video_tokens: self.max_v_len
            mask: self.max_v_len
        """
        max_v_l = self.max_v_len - 2
        feat_len = len(raw_feat)
        st, ed = clip_st_ed
        ed = min(ed, feat_len-1)
        if st > ed:
            st = max(0, ed-1)
        indexed_feat_len = ed - st + 1

        feat = np.zeros((self.max_v_len + self.max_sub_len, raw_feat.shape[1]))  # includes [CLS], [SEP]
        if indexed_feat_len > max_v_l:
            downsamlp_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(np.int64).tolist()
            assert max(downsamlp_indices) < feat_len
            if self.normalize_vfeat:
                feat[1:max_v_l+1] = l2_normalize_np_array(raw_feat[downsamlp_indices])  # truncate, sample???
            else:
                feat[1:max_v_l + 1] = raw_feat[downsamlp_indices]

            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * max_v_l + [self.SEP_TOKEN]
            mask = [1] * (max_v_l + 2)
        else:
            valid_l = ed - st + 1
            feat[1:valid_l+1] = raw_feat[st:ed + 1]
            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * valid_l + \
                           [self.SEP_TOKEN] + [self.PAD_TOKEN] * (max_v_l - valid_l)
            mask = [1] * (valid_l + 2) + [0] * (max_v_l - valid_l)
        return feat, video_tokens, mask

    def get_context(self, example):
        """example single snetence
        {"vid_name": str,
         "duration": float,
         "ts": [st(float), ed(float)],
         "desc": str,
         "clip_id": str
        }
        """
        vid_name = example["vid_name"]
        clip_st_ed = example["clip_st_ed"]

        # video + text tokens
        # [CLS] [VID], ..., [VID], [PAD], ..., [PAD], [SEP], [BOS], [WORD], ..., [WORD], [EOS], [PAD], ...
        if self.use_video:
            video_feature = self.vid_h5[vid_name]  # (L, D)
            video_feat, video_tokens, video_mask = self._load_indexed_video_feature(video_feature, clip_st_ed)
        else:
            video_feat, video_tokens, video_mask = np.zeros((2+self.max_sub_len, 2)), [0, 0], [0, 0]  # fake inputs
        if self.use_sub:
            sub_tokens, sub_mask = self._load_indexed_sub(self.sub_meta_dict[vid_name], clip_st_ed)
        else:
            sub_tokens, sub_mask = [0, 0], [0, 0]

        ctx_input_tokens = video_tokens + sub_tokens

        ctx_input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in ctx_input_tokens]
        ctx_input_mask = video_mask + sub_mask
        ctx_token_type_ids = [0] * self.max_v_len + [1] * self.max_sub_len
        return ctx_input_ids, ctx_input_mask, ctx_token_type_ids, video_feat, video_mask

    def get_caption(self, example):
        """example: """
        caption_tokens, caption_mask = self._tokenize_and_pad_sentence(example["desc"], max_sen_l=self.max_cap_len)
        caption_input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in caption_tokens]
        return caption_input_ids, caption_mask

    def convert_ts_to_clip_indices(self, ts):
        return [int(math.floor(ts[0] / self.clip_length)), int(math.ceil(ts[1] / self.clip_length))]

    def convert_ids_to_sentence(self, ids, rm_padding=True, return_sentence_only=True):
        """A list of token ids"""
        rm_padding = True if return_sentence_only else rm_padding
        if rm_padding:
            raw_words = [self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]]
        else:
            raw_words = [self.idx2word[wid] for wid in ids if wid != self.IGNORE]

        # get only sentences, the tokens between `[BOS]` and the first `[EOS]`
        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.EOS_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        return " ".join(words)


def prepare_batch_inputs(batch, device, non_blocking=False):
    batch_inputs = dict()
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_inputs[k] = v.to(device, non_blocking=non_blocking)
        else:  # all non-tensor values
            batch_inputs[k] = v
    return batch_inputs


def step_collate(padded_batch_step):
    """The same step (clip-sentence pair) from each example"""
    c_batch = dict()
    for key in padded_batch_step[0]:
        value = padded_batch_step[0][key]
        if isinstance(value, list):
            c_batch[key] = [d[key] for d in padded_batch_step]
        else:
            c_batch[key] = default_collate([d[key] for d in padded_batch_step])
    return c_batch


def caption_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66
    """
    # collect meta
    batch_meta = [e[1] for e in batch]  # change key
    model_inputs = step_collate([e[0] for e in batch])
    return model_inputs, batch_meta


def process_single_vid_sub(sub_listdicts, clip_length):
    """
    Args:
        sub_listdicts: list(dicts), each dict is, e.g.,
            {'text': " Chase : That's all this is?", 'start': 0.862, 'end': 1.862}
        clip_length: float
    Returns:
        clip_idx2sentence_indices: dict, {clip_idx: [sen_idx1, sen_idx2, ...]}, which sentences are
            associated with which clips. The indices are in ascending order, i.e., sen_idx1 < sen_idx2 < ...
    """
    timestamps = np.array([[e["start"], e["end"]] for e in sub_listdicts], dtype=np.float32)  # (n_sub_sen, 2)
    timestamps = timestamps / clip_length
    # r-th row of clip_indices is [st_idx, ed_idx), where [st_idx, st_idx+1, ..., ed_idx-1]
    # should be with r-th clip, which is [r*clip_length, (r+1)*clip_length]
    sentence2clip_st_ed = np.empty_like(timestamps, dtype=np.int64)
    sentence2clip_st_ed[:, 0] = np.floor(timestamps[:, 0])
    sentence2clip_st_ed[:, 1] = np.ceil(timestamps[:, 1])
    sentence_idx2clip_indices = {sen_idx: set(range(clip_st_idx, clip_ed_idx))
                                 for sen_idx, (clip_st_idx, clip_ed_idx) in enumerate(sentence2clip_st_ed)}
    all_clip_indices = set(flat_list_of_lists(list(sentence_idx2clip_indices.values())))
    clip_idx2sentence_indices = \
        {str(clip_idx): sorted([k for k, v in sentence_idx2clip_indices.items() if clip_idx in v])
         for clip_idx in all_clip_indices}
    return clip_idx2sentence_indices


def load_process_sub_meta(sub_meta_path, clip_length):
    """ which subtitle sentences should be assigned to which clips
    Args:
        sub_meta_path: contains a jsonl file, each line is a dict {"vid_name": str, "sub": list(dicts)},
            each dict under "sub" is, e.g., {'text': " Chase : That's all this is?", 'start': 0.862, 'end': 1.862}.
            The dicts under "sub" are ordered the same as the original .srt files.
        clip_length: float, assign each subtitle sentence to a clip segment
    Returns:
    """
    video2sub = {e["vid_name"]: e for e in load_jsonl(sub_meta_path)}
    for vid_name, sub_info in tqdm(video2sub.items(), desc="processing subtitles"):
        sub_info["clip2sen"] = process_single_vid_sub(sub_info["sub"], clip_length)
        video2sub[vid_name] = sub_info
    return video2sub
