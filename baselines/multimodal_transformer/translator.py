""" This module will handle the text generation with beam search. """

import torch
import torch.nn.functional as F

from baselines.multimodal_transformer.transformer.model import MMT
from baselines.multimodal_transformer.transformer.beam_search import BeamSearch
from baselines.multimodal_transformer.transformer.tvc_dataset import TVCaptionDataset

import logging
logger = logging.getLogger(__name__)


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def mask_tokens_after_eos(input_ids, input_masks,
                          eos_token_id=TVCaptionDataset.EOS, pad_token_id=TVCaptionDataset.PAD):
    """replace values after `[EOS]` with `[PAD]`,
    used to compute memory for next sentence generation"""
    for row_idx in range(len(input_ids)):
        # possibly more than one `[EOS]`
        cur_eos_idxs = (input_ids[row_idx] == eos_token_id).nonzero()
        if len(cur_eos_idxs) != 0:
            cur_eos_idx = cur_eos_idxs[0, 0].item()
            input_ids[row_idx, cur_eos_idx+1:] = pad_token_id
            input_masks[row_idx, cur_eos_idx+1:] = 0
    return input_ids, input_masks


class Translator(object):
    """Load with trained model and handle the beam search"""
    def __init__(self, opt, checkpoint, model=None):
        self.opt = opt
        self.device = opt.device

        self.max_cap_len = opt.max_cap_len

        if model is None:
            model = MMT(checkpoint["model_cfg"]).to(self.device)
            model.load_state_dict(checkpoint["model"])
        print("[Info] Trained model state loaded.")
        self.model = model
        self.model.eval()

    def translate_batch_beam(self, model_inputs, model,
                             beam_size, n_best, min_length, max_length, block_ngram_repeat, exclusion_idxs,
                             device, length_penalty_name, length_penalty_alpha):
        beam = BeamSearch(
            beam_size,
            n_best=n_best,
            batch_size=len(model_inputs["ctx_input_ids"]),
            pad=TVCaptionDataset.PAD,
            eos=TVCaptionDataset.EOS,
            bos=TVCaptionDataset.BOS,
            min_length=min_length,
            max_length=max_length,
            mb_device=device,
            block_ngram_repeat=block_ngram_repeat,
            exclusion_tokens=exclusion_idxs,
            length_penalty_name=length_penalty_name,
            length_penalty_alpha=length_penalty_alpha
        )

        def duplicate_for_beam(model_inputs, beam_size):
            for k, v in model_inputs.items():
                model_inputs[k] = tile(v, beam_size, dim=0)  # (N * beam_size, L)
            return model_inputs

        encoder_outputs = model.encode(model_inputs["ctx_input_ids"],
                                       model_inputs["ctx_input_mask"],
                                       model_inputs["ctx_token_type_ids"],
                                       model_inputs["video_feature"])  # (N, Lv, D)
        model_inputs = dict(
            encoder_outputs=encoder_outputs,
            ctx_input_mask=model_inputs["ctx_input_mask"]
        )
        model_inputs = duplicate_for_beam(model_inputs, beam_size=beam_size)

        bsz = len(model_inputs["encoder_outputs"])
        text_input_ids = model_inputs["encoder_outputs"].new_zeros(bsz, max_length).long()  # zeros
        text_masks = model_inputs["ctx_input_mask"].new_zeros(bsz, max_length)  # zeros
        encoder_outputs = model_inputs["encoder_outputs"]
        encoder_masks = model_inputs["ctx_input_mask"]

        for dec_idx in range(max_length):
            text_input_ids[:, dec_idx] = beam.current_predictions
            text_masks[:, dec_idx] = 1
            _, pred_scores = model.decode(
                text_input_ids, text_masks, encoder_outputs, encoder_masks, text_input_labels=None)

            pred_scores[:, TVCaptionDataset.UNK] = -1e10  # remove `[UNK]` token
            logprobs = torch.log(F.softmax(pred_scores[:, dec_idx], dim=1))  # (N * beam_size, vocab_size)
            beam.advance(logprobs)
            any_beam_is_finished = beam.is_finished.any()
            if any_beam_is_finished:
                beam.update_finished()
                if beam.done:
                    break

            if any_beam_is_finished:
                # update input args
                select_indices = beam.current_origin  # N * B, i.e. batch_size * beam_size
                text_input_ids = text_input_ids.index_select(0, select_indices)
                text_masks = text_masks.index_select(0, select_indices)
                encoder_outputs = encoder_outputs.index_select(0, select_indices)
                encoder_masks = encoder_masks.index_select(0, select_indices)

        # fill in generated words
        text_input_ids = model_inputs["encoder_outputs"].new_zeros(bsz, max_length).long()  # zeros
        text_masks = model_inputs["ctx_input_mask"].new_zeros(bsz, max_length)  # zero
        for batch_idx in range(len(beam.predictions)):
            cur_sen_ids = beam.predictions[batch_idx][0].cpu().tolist()  # use the top sentences
            cur_sen_ids = [TVCaptionDataset.BOS] + cur_sen_ids + [TVCaptionDataset.EOS]
            cur_sen_len = len(cur_sen_ids)
            text_input_ids[batch_idx, :cur_sen_len] = text_input_ids.new(cur_sen_ids)
            text_masks[batch_idx, :cur_sen_len] = 1

        # compute memory, mimic the way memory is generated at training time
        text_input_ids, text_masks = mask_tokens_after_eos(text_input_ids, text_masks)
        return text_input_ids

    @classmethod
    def translate_batch_single_sentence_untied_greedy(
            cls,
            model_inputs,
            model,
            start_idx=TVCaptionDataset.BOS,
            unk_idx=TVCaptionDataset.UNK,
            max_cap_len=None):
        """The first few args are the same to the input to the forward_step func
        Note:
            1, Copy the prev_ms each word generation step, as the func will modify this value,
            which will cause discrepancy between training and inference
            2, After finish the current sentence generation step, replace the words generated
            after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
            next memory state tensor.
        """
        encoder_outputs = model.encode(model_inputs["ctx_input_ids"],
                                       model_inputs["ctx_input_mask"],
                                       model_inputs["ctx_token_type_ids"],
                                       model_inputs["video_feature"])  # (N, Lv, D)

        bsz = len(model_inputs["ctx_input_ids"])
        max_cap_len = max_cap_len
        text_input_ids = model_inputs["ctx_input_ids"].new_zeros(bsz, max_cap_len)  # zeros
        text_masks = model_inputs["ctx_input_ids"].new_zeros(bsz, max_cap_len).float()  # zeros
        next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
        for dec_idx in range(max_cap_len):
            text_input_ids[:, dec_idx] = next_symbols
            text_masks[:, dec_idx] = 1
            _, pred_scores = model.decode(
                text_input_ids, text_masks, encoder_outputs, model_inputs["ctx_input_mask"], text_input_labels=None)
            # suppress unk token; (N, L, vocab_size)
            pred_scores[:, :, unk_idx] = -1e10
            next_words = pred_scores[:, dec_idx].max(1)[1]
            next_symbols = next_words
        return text_input_ids  # (N, Lt)

    def translate_batch(self, model_inputs, max_cap_len, use_beam=False):
        """while we used *_list as the input names, they could be non-list for single sentence decoding case"""
        if use_beam:
            return self.translate_batch_beam(model_inputs, self.model,
                                             beam_size=self.opt.beam_size, n_best=self.opt.n_best,
                                             min_length=self.opt.min_sen_len, max_length=self.opt.max_sen_len - 2,
                                             block_ngram_repeat=self.opt.block_ngram_repeat, exclusion_idxs=[],
                                             device=self.device,
                                             length_penalty_name=self.opt.length_penalty_name,
                                             length_penalty_alpha=self.opt.length_penalty_alpha)
        else:
            return self.translate_batch_single_sentence_untied_greedy(
                model_inputs, self.model, max_cap_len=max_cap_len)
