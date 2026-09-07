import torch
import random
import numpy as np
def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]

import torch.nn.functional as F
DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

#applying connection metric only for text tokens but taking text-text approach

class ModifiedElasticCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.0,
        distance=-25,
        layer_num=40,
        attn_weight=1.0,
        conn_weight=0.8,
        image_start=None,
        image_token_len=576,
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

        self.score_sum = torch.zeros(layer_num, self.cache_size + 1).cuda()
        self.conn_score_sum = torch.zeros(layer_num, self.cache_size + 1).cuda()

        self.ratio = ratio
        self.protect_size = 1
        self.flag = True
        self.distance = distance
        self.layer_num = layer_num

        self.attn_weight = attn_weight
        self.conn_weight = conn_weight

        self.image_start = image_start
        self.image_token_len = image_token_len

        self.selected_idx = 0

    def _build_modality_masks(self, seq_len, device):
        text_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        image_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

        if self.image_start is None:
            return text_mask, image_mask

        image_end = min(self.image_start + self.image_token_len, seq_len)

        if self.image_start < seq_len:
            text_mask[self.image_start:image_end] = False
            image_mask[self.image_start:image_end] = True

        return text_mask, image_mask

    def reset(self):
        self.score_sum.zero_()
        self.conn_score_sum.zero_()
        self.flag = True
        self.selected_idx = 0

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None

        attn_score = [attention for attention in attentions]
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        attn_score = torch.cat(attn_score, dim=0)
        attn_score = attn_score.mean(dim=1, keepdim=False)

        text_mask, image_mask = self._build_modality_masks(seq_len, attn_score.device)

        if attn_score.shape[-2] > 1:
            if self.flag is not True:
                return past_key_values

            for idx in range(attn_score.shape[-1]):
                cur_score = attn_score[:, idx, :idx + 1]

                self.score_sum[:, :cur_score.shape[-1]] += cur_score

                if text_mask[idx].item():
                    conn_score = cur_score * image_mask[:idx + 1].unsqueeze(0).float()
                else:
                    conn_score = cur_score * text_mask[:idx + 1].unsqueeze(0).float()


                self.conn_score_sum[:, :cur_score.shape[-1]] += conn_score

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))

        if forget_num <= 0:
            return past_key_values

        if forget_num > 1:
            if self.flag is not True:
                return past_key_values
            self.flag = False

            selected_idx_all = []
            merge_idx_all = []
            throw_idx_all = []

            for layer_idx in range(self.layer_num):
                start = self.start_size
                end = seq_len - self.protect_size

                raw_score = self.score_sum[layer_idx, start:end]
                conn_score = self.conn_score_sum[layer_idx, start:end]

                local_text_mask = text_mask[start:end]

                raw_score_norm = raw_score / (raw_score.max() + 1e-6)

                conn_score_norm = torch.zeros_like(conn_score)
                if local_text_mask.any():
                    conn_text = conn_score[local_text_mask]
                    conn_score_norm[local_text_mask] = conn_text / (conn_text.max() + 1e-6)

                combined_score = self.attn_weight * raw_score_norm
                combined_score[local_text_mask] += (
                    self.conn_weight * conn_score_norm[local_text_mask]
                )

                order = torch.argsort(combined_score)
                throw_idx = order[:forget_num] + start
                selected_idx = order[forget_num:] + start

                merge_idx = []
                for i in range(len(throw_idx)):
                    nearest_idx = torch.abs(selected_idx - throw_idx[i]).argmin()
                    merge_idx.append(selected_idx[nearest_idx].unsqueeze(0))
                merge_idx = torch.cat(merge_idx)

                selected_idx = torch.cat(
                    [
                        torch.arange(self.start_size).cuda(),
                        selected_idx,
                        torch.tensor([seq_len - self.protect_size]).cuda(),
                    ],
                    dim=0
                )

                selected_idx_all.append(selected_idx)
                merge_idx_all.append(merge_idx)
                throw_idx_all.append(throw_idx)

            if self.distance > 0:
                self.selected_idx = self.distance
            else:
                self.selected_idx = seq_len - forget_num + self.distance

            past_key_values_return = []

            for layer_idx, (k, v) in enumerate(past_key_values):
                selected_idx = selected_idx_all[layer_idx]
                merge_idx = merge_idx_all[layer_idx]
                throw_idx = throw_idx_all[layer_idx]

                k_forget = k.gather(
                    dim=-2,
                    index=throw_idx.view(1, 1, -1, 1).expand(
                        k.shape[0], k.shape[1], -1, k.shape[-1]
                    )
                )
                v_forget = v.gather(
                    dim=-2,
                    index=throw_idx.view(1, 1, -1, 1).expand(
                        v.shape[0], v.shape[1], -1, v.shape[-1]
                    )
                )

                k = k.scatter_reduce(
                    -2,
                    merge_idx.view(1, 1, -1, 1).expand(
                        k.shape[0], k.shape[1], -1, k.shape[-1]
                    ),
                    k_forget,
                    "mean"
                )
                v = v.scatter_reduce(
                    -2,
                    merge_idx.view(1, 1, -1, 1).expand(
                        v.shape[0], v.shape[1], -1, v.shape[-1]
                    ),
                    v_forget,
                    "mean"
                )

                k_new = k.gather(
                    dim=-2,
                    index=selected_idx.view(1, 1, -1, 1).expand(
                        k.shape[0], k.shape[1], -1, k.shape[-1]
                    )
                )
                v_new = v.gather(
                    dim=-2,
                    index=selected_idx.view(1, 1, -1, 1).expand(
                        v.shape[0], v.shape[1], -1, v.shape[-1]
                    )
                )

                past_key_values_return.append([k_new, v_new])

            return past_key_values_return

        else:
            selected_idx = self.selected_idx
            return [[
                torch.cat(
                    [
                        self.k_slice(k, 0, selected_idx),
                        self.k_slice(k, selected_idx + 1, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, selected_idx),
                        self.v_slice(v, selected_idx + 1, seq_len),
                    ],
                    dim=self.v_seq_dim,
                )
            ] for k, v in past_key_values]

#using global and recent attention score
# class ElasticCache:
#     def __init__(
#         self,
#         start_size=4,
#         recent_size=512,
#         k_seq_dim=2,
#         v_seq_dim=2,
#         ratio=0.0,
#         distance=-25,
#         layer_num=40,
#         global_weight=1.0,     # weight for long-term/global importance
#         recent_weight=0.1,     # weight for short-term/recent importance
#         recent_window=128,     # number of recent decoding/query steps to keep in recent score
#     ):
#         self.start_size = start_size
#         self.recent_size = recent_size
#         self.cache_size = start_size + recent_size

#         self.k_seq_dim = k_seq_dim
#         self.v_seq_dim = v_seq_dim
#         self.k_slice = DIM_TO_SLICE[k_seq_dim]
#         self.v_slice = DIM_TO_SLICE[v_seq_dim]

#         # Global accumulated attention importance
#         self.score_sum = torch.zeros(layer_num, self.cache_size + 1).cuda()

#         # NEW: recent-window accumulated attention importance
#         self.recent_score_sum = torch.zeros(layer_num, self.cache_size + 1).cuda()

#         self.ratio = ratio
#         self.protect_size = 1
#         self.flag = True
#         self.distance = distance
#         self.layer_num = layer_num

#         # NEW: weights for hybrid importance
#         self.global_weight = global_weight
#         self.recent_weight = recent_weight

#         # NEW: recent attention window size
#         self.recent_window = recent_window

#         # NEW: store per-step contributions for sliding recent window
#         self.recent_history = []

#         self.selected_idx = 0

#     def _maybe_expand_score_buffers(self, seq_len, device):
#         """
#         Expand score buffers if current seq_len is larger than allocated size.
#         """
#         cur_len = self.score_sum.shape[1]
#         if seq_len <= cur_len:
#             return

#         new_score_sum = torch.zeros(self.layer_num, seq_len, device=device)
#         new_recent_score_sum = torch.zeros(self.layer_num, seq_len, device=device)

#         new_score_sum[:, :cur_len] = self.score_sum
#         new_recent_score_sum[:, :cur_len] = self.recent_score_sum

#         self.score_sum = new_score_sum
#         self.recent_score_sum = new_recent_score_sum

#     def reset(self):
#         """
#         Optional helper to reset internal state between samples.
#         """
#         self.score_sum.zero_()
#         self.recent_score_sum.zero_()
#         self.recent_history = []
#         self.flag = True
#         self.selected_idx = 0

#     def __call__(self, past_key_values, num_of_token=None, attentions=None):
#         if past_key_values is None:
#             return None

#         seq_len = past_key_values[0][0].size(self.k_seq_dim)
#         device = past_key_values[0][0].device

#         self._maybe_expand_score_buffers(seq_len, device)

#         # attentions: list of [batch, heads, q_len, k_len] per layer
#         attn_score = [attention for attention in attentions]
#         attn_score = torch.cat(attn_score, dim=0)           # [layer, heads, q, k] if batch=1
#         attn_score = attn_score.mean(dim=1, keepdim=False)  # average over heads -> [layer, q, k]

#         # Update scores before first major compression
#         if attn_score.shape[-2] > 1:
#             assert self.flag is True

#             # Each query position contributes attention to all keys up to itself
#             for idx in range(attn_score.shape[-1]):
#                 cur_score = attn_score[:, idx, :idx + 1]  # [layer, idx+1]

#                 # -------------------------
#                 # GLOBAL SCORE UPDATE
#                 # -------------------------
#                 self.score_sum[:, :cur_score.shape[-1]] += cur_score

#                 # -------------------------
#                 # RECENT SCORE UPDATE
#                 # -------------------------
#                 # Store the per-query contribution as one history entry
#                 contrib = torch.zeros(self.layer_num, seq_len, device=device)
#                 contrib[:, :cur_score.shape[-1]] = cur_score

#                 self.recent_score_sum[:, :seq_len] += contrib
#                 self.recent_history.append(contrib)

#                 # keep only last recent_window query contributions
#                 if len(self.recent_history) > self.recent_window:
#                     old_contrib = self.recent_history.pop(0)
#                     self.recent_score_sum[:, :old_contrib.shape[-1]] -= old_contrib

#         forget_num = int(seq_len - num_of_token * (1 - self.ratio))

#         if forget_num <= 0:
#             return past_key_values

#         if forget_num > 1:
#             assert self.flag is True
#             self.flag = False

#             selected_idx_all = []
#             merge_idx_all = []
#             throw_idx_all = []

#             for layer_idx in range(self.layer_num):
#                 start = self.start_size
#                 end = seq_len - self.protect_size

#                 global_score = self.score_sum[layer_idx, start:end]
#                 recent_score = self.recent_score_sum[layer_idx, start:end]

#                 # Normalize both so one score does not dominate only because of scale
#                 global_score_norm = global_score / (global_score.max() + 1e-6)
#                 recent_score_norm = recent_score / (recent_score.max() + 1e-6)

#                 # -------------------------
#                 # HYBRID IMPORTANCE
#                 # -------------------------
#                 combined_score = (
#                     self.global_weight * global_score_norm +
#                     self.recent_weight * recent_score_norm
#                 )

#                 # low score -> remove, high score -> keep
#                 order = torch.argsort(combined_score)  # ascending
#                 throw_idx = order[:forget_num] + start
#                 selected_idx = order[forget_num:] + start

#                 # merge each removed token into nearest kept token
#                 merge_idx = []
#                 for i in range(len(throw_idx)):
#                     nearest_idx = torch.abs(selected_idx - throw_idx[i]).argmin()
#                     merge_idx.append(selected_idx[nearest_idx].unsqueeze(0))
#                 merge_idx = torch.cat(merge_idx)

#                 # always keep prefix and last token
#                 selected_idx = torch.cat(
#                     [
#                         torch.arange(self.start_size, device=device),
#                         selected_idx,
#                         torch.tensor([seq_len - self.protect_size], device=device),
#                     ],
#                     dim=0
#                 )

#                 selected_idx_all.append(selected_idx)
#                 merge_idx_all.append(merge_idx)
#                 throw_idx_all.append(throw_idx)

#             if self.distance > 0:
#                 self.selected_idx = self.distance
#             else:
#                 self.selected_idx = seq_len - forget_num + self.distance

#             past_key_values_return = []

#             for layer_idx, (k, v) in enumerate(past_key_values):
#                 selected_idx = selected_idx_all[layer_idx]
#                 merge_idx = merge_idx_all[layer_idx]
#                 throw_idx = throw_idx_all[layer_idx]

#                 k_forget = k.gather(
#                     dim=-2,
#                     index=throw_idx.view(1, 1, -1, 1).expand(
#                         k.shape[0], k.shape[1], -1, k.shape[-1]
#                     )
#                 )
#                 v_forget = v.gather(
#                     dim=-2,
#                     index=throw_idx.view(1, 1, -1, 1).expand(
#                         v.shape[0], v.shape[1], -1, v.shape[-1]
#                     )
#                 )

#                 k = k.scatter_reduce(
#                     -2,
#                     merge_idx.view(1, 1, -1, 1).expand(
#                         k.shape[0], k.shape[1], -1, k.shape[-1]
#                     ),
#                     k_forget,
#                     "mean"
#                 )
#                 v = v.scatter_reduce(
#                     -2,
#                     merge_idx.view(1, 1, -1, 1).expand(
#                         v.shape[0], v.shape[1], -1, v.shape[-1]
#                     ),
#                     v_forget,
#                     "mean"
#                 )

#                 k_new = k.gather(
#                     dim=-2,
#                     index=selected_idx.view(1, 1, -1, 1).expand(
#                         k.shape[0], k.shape[1], -1, k.shape[-1]
#                     )
#                 )
#                 v_new = v.gather(
#                     dim=-2,
#                     index=selected_idx.view(1, 1, -1, 1).expand(
#                         v.shape[0], v.shape[1], -1, v.shape[-1]
#                     )
#                 )

#                 past_key_values_return.append([k_new, v_new])

#             return past_key_values_return

#         else:
#             # incremental single-token removal after first compression
#             selected_idx = self.selected_idx
#             return [[
#                 torch.cat(
#                     [
#                         self.k_slice(k, 0, selected_idx),
#                         self.k_slice(k, selected_idx + 1, seq_len),
#                     ],
#                     dim=self.k_seq_dim,
#                 ),
#                 torch.cat(
#                     [
#                         self.v_slice(v, 0, selected_idx),
#                         self.v_slice(v, selected_idx + 1, seq_len),
#                     ],
#                     dim=self.v_seq_dim,
#                 )
#             ] for k, v in past_key_values]

#cross-modal interation for importance 
# class ElasticCache:
#     def __init__(
#         self,
#         start_size=4,              # always keep first few tokens
#         recent_size=512,           # target cache budget
#         k_seq_dim=2,               # sequence dimension in K
#         v_seq_dim=2,               # sequence dimension in V
#         ratio=0.0,                 # forgetting aggressiveness
#         distance=-25,              # used in incremental single-token removal
#         layer_num=40,              # number of transformer layers
#         attn_weight=1.0,           # weight for original attention score
#         cross_modal_weight=0.3,    # weight for cross-modal score
#         image_start=None,          # where image token block starts in the sequence
#         image_token_len=576,       # length of image token block
#     ):
#         self.start_size = start_size
#         self.recent_size = recent_size
#         self.cache_size = start_size + recent_size

#         self.k_seq_dim = k_seq_dim
#         self.v_seq_dim = v_seq_dim
#         self.k_slice = DIM_TO_SLICE[k_seq_dim]
#         self.v_slice = DIM_TO_SLICE[v_seq_dim]

    #     # accumulated original attention importance
    #     self.score_sum = torch.zeros(layer_num, self.cache_size + 1).cuda()

    #     # accumulated cross-modal score
    #     self.cross_modal_score_sum = torch.zeros(layer_num, self.cache_size + 1).cuda()

    #     self.ratio = ratio
    #     self.protect_size = 1      # always keep last token
    #     self.flag = True           # main compression block runs once
    #     self.distance = distance
    #     self.layer_num = layer_num

    #     self.attn_weight = attn_weight
    #     self.cross_modal_weight = cross_modal_weight

    #     self.image_start = image_start
    #     self.image_token_len = image_token_len

    #     self.selected_idx = 0

    # def _build_modality_masks(self, seq_len, device):
    #     """
    #     Build text/image masks for a unified multimodal sequence.

    #     Assumed layout:
    #         [text before image][image block][text after image][generated text...]

    #     If image_start is not set, treat everything as text.
    #     """
    #     text_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    #     image_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

    #     if self.image_start is None:
    #         return text_mask, image_mask

    #     image_end = min(self.image_start + self.image_token_len, seq_len)

    #     if self.image_start < seq_len:
    #         text_mask[self.image_start:image_end] = False
    #         image_mask[self.image_start:image_end] = True

    #     return text_mask, image_mask

  

    # def _maybe_expand_score_buffers(self, seq_len, device):
    #     """
    #     Expand score buffers if current sequence is longer than allocated size.
    #     """
    #     cur_len = self.score_sum.shape[1]
    #     if seq_len <= cur_len:
    #         return

    #     new_score_sum = torch.zeros(self.layer_num, seq_len, device=device)
    #     new_cross_sum = torch.zeros(self.layer_num, seq_len, device=device)

    #     new_score_sum[:, :cur_len] = self.score_sum
    #     new_cross_sum[:, :cur_len] = self.cross_modal_score_sum

    #     self.score_sum = new_score_sum
    #     self.cross_modal_score_sum = new_cross_sum

    # def __call__(self, past_key_values, num_of_token=None, attentions=None):
    #     if past_key_values is None:
    #         return None

    #     seq_len = past_key_values[0][0].size(self.k_seq_dim)
    #     device = past_key_values[0][0].device

    #     self._maybe_expand_score_buffers(seq_len, device)

    #     # attentions: list of layer attentions
    #     # expected shape per layer: [batch, heads, q_len, k_len]
    #     attn_score = [attention for attention in attentions]
    #     attn_score = torch.cat(attn_score, dim=0)                  # [layer, heads, q, k] if batch=1
    #     attn_score = attn_score.mean(dim=1, keepdim=False)         # average over heads -> [layer, q, k]

    #     text_mask, image_mask = self._build_modality_masks(seq_len, attn_score.device)

    #     # update scores only before first major compression
    #     if attn_score.shape[-2] > 1:
    #         assert self.flag is True

    #         for idx in range(attn_score.shape[-1]):
    #             # attention from query idx to valid keys [0 ... idx]
    #             cur_score = attn_score[:, idx, :idx + 1]           # [layer, idx+1]

    #             # original metric: accumulate raw attention
    #             self.score_sum[:, :cur_score.shape[-1]] += cur_score

    #             # cross-modal metric:
    #             # if query is text, count only attention to image tokens
    #             # if query is image, count only attention to text tokens
    #             cur_text_mask = text_mask[:idx + 1]
    #             cur_image_mask = image_mask[:idx + 1]

    #             if text_mask[idx].item():
    #                 cross_score = cur_score * cur_image_mask.unsqueeze(0).float()
    #             else:
    #                 cross_score = cur_score * cur_text_mask.unsqueeze(0).float()

    #             self.cross_modal_score_sum[:, :cur_score.shape[-1]] += cross_score

    #     forget_num = int(seq_len - num_of_token * (1 - self.ratio))

    #     if forget_num <= 0:
    #         return past_key_values

    #     if forget_num > 1:
    #         assert self.flag is True
    #         self.flag = False

    #         selected_idx_all = []
    #         merge_idx_all = []
    #         throw_idx_all = []

    #         for layer_idx in range(self.layer_num):
    #             start = self.start_size
    #             end = seq_len - self.protect_size

    #             raw_score = self.score_sum[layer_idx, start:end]
    #             cross_score = self.cross_modal_score_sum[layer_idx, start:end]

    #             local_text_mask = text_mask[start:end]
    #             local_image_mask = image_mask[start:end]

    #             # modality-aware normalization
    #             raw_score_norm = torch.zeros_like(raw_score)
    #             cross_score_norm = torch.zeros_like(cross_score)

    #             if local_text_mask.any():
    #                 raw_text = raw_score[local_text_mask]
    #                 cross_text = cross_score[local_text_mask]

    #                 raw_score_norm[local_text_mask] = raw_text / (raw_text.max() + 1e-6)
    #                 cross_score_norm[local_text_mask] = cross_text / (cross_text.max() + 1e-6)

    #             if local_image_mask.any():
    #                 raw_image = raw_score[local_image_mask]
    #                 cross_image = cross_score[local_image_mask]

    #                 raw_score_norm[local_image_mask] = raw_image / (raw_image.max() + 1e-6)
    #                 cross_score_norm[local_image_mask] = cross_image / (cross_image.max() + 1e-6)

    #             # combined importance
    #             combined_score = (
    #                 self.attn_weight * raw_score_norm +
    #                 self.cross_modal_weight * cross_score_norm
    #             )

    #             # low score -> remove, high score -> keep
    #             order = torch.argsort(combined_score)
    #             throw_idx = order[:forget_num] + start
    #             selected_idx = order[forget_num:] + start

    #             # map each removed token to nearest kept token
    #             merge_idx = []
    #             for i in range(len(throw_idx)):
    #                 nearest_idx = torch.abs(selected_idx - throw_idx[i]).argmin()
    #                 merge_idx.append(selected_idx[nearest_idx].unsqueeze(0))
    #             merge_idx = torch.cat(merge_idx)

    #             # always keep prefix + selected + last token
    #             selected_idx = torch.cat(
    #                 [
    #                     torch.arange(self.start_size, device=device),
    #                     selected_idx,
    #                     torch.tensor([seq_len - self.protect_size], device=device),
    #                 ],
    #                 dim=0
    #             )

    #             selected_idx_all.append(selected_idx)
    #             merge_idx_all.append(merge_idx)
    #             throw_idx_all.append(throw_idx)

    #         if self.distance > 0:
    #             self.selected_idx = self.distance
    #         else:
    #             self.selected_idx = seq_len - forget_num + self.distance

    #         past_key_values_return = []

    #         for layer_idx, (k, v) in enumerate(past_key_values):
    #             selected_idx = selected_idx_all[layer_idx]
    #             merge_idx = merge_idx_all[layer_idx]
    #             throw_idx = throw_idx_all[layer_idx]

    #             k_forget = k.gather(
    #                 dim=-2,
    #                 index=throw_idx.view(1, 1, -1, 1).expand(
    #                     k.shape[0], k.shape[1], -1, k.shape[-1]
    #                 )
    #             )
    #             v_forget = v.gather(
    #                 dim=-2,
    #                 index=throw_idx.view(1, 1, -1, 1).expand(
    #                     v.shape[0], v.shape[1], -1, v.shape[-1]
    #                 )
    #             )

    #             k = k.scatter_reduce(
    #                 -2,
    #                 merge_idx.view(1, 1, -1, 1).expand(
    #                     k.shape[0], k.shape[1], -1, k.shape[-1]
    #                 ),
    #                 k_forget,
    #                 "mean"
    #             )
    #             v = v.scatter_reduce(
    #                 -2,
    #                 merge_idx.view(1, 1, -1, 1).expand(
    #                     v.shape[0], v.shape[1], -1, v.shape[-1]
    #                 ),
    #                 v_forget,
    #                 "mean"
    #             )

    #             k_new = k.gather(
    #                 dim=-2,
    #                 index=selected_idx.view(1, 1, -1, 1).expand(
    #                     k.shape[0], k.shape[1], -1, k.shape[-1]
    #                 )
    #             )
    #             v_new = v.gather(
    #                 dim=-2,
    #                 index=selected_idx.view(1, 1, -1, 1).expand(
    #                     v.shape[0], v.shape[1], -1, v.shape[-1]
    #                 )
    #             )

    #             past_key_values_return.append([k_new, v_new])

    #         return past_key_values_return

    #     else:
    #         # incremental removal: remove one token using saved index
    #         selected_idx = self.selected_idx
    #         return [[
    #             torch.cat(
    #                 [
    #                     self.k_slice(k, 0, selected_idx),
    #                     self.k_slice(k, selected_idx + 1, seq_len),
    #                 ],
    #                 dim=self.k_seq_dim,
    #             ),
    #             torch.cat(
    #                 [
    #                     self.v_slice(v, 0, selected_idx),
    #                     self.v_slice(v, selected_idx + 1, seq_len),
    #                 ],
    #                 dim=self.v_seq_dim,
    #             )
    #         ] for k, v in past_key_values]

#normalized connection values
# class ElasticCache:
#     def __init__(
#         self,
#         start_size=4,
#         recent_size=512,
#         k_seq_dim=2,
#         v_seq_dim=2,
#         ratio=0.,
#         distance=-25,
#         layer_num=40,
#         attn_weight=1.0,
#         cross_modal_weight=0.5,
#         text_token_count=None,      # NEW: number of text tokens in the prefix of sequence
#     ):
#         self.start_size = start_size
#         self.recent_size = recent_size
#         self.cache_size = start_size + recent_size
#         self.k_seq_dim = k_seq_dim
#         self.v_seq_dim = v_seq_dim
#         self.k_slice = DIM_TO_SLICE[k_seq_dim]
#         self.v_slice = DIM_TO_SLICE[v_seq_dim]

#         # Original accumulated attention importance
#         self.score_sum = torch.zeros(layer_num, self.cache_size + 1).cuda()

#         # NEW: accumulated cross-modal relation score
#         self.cross_modal_score_sum = torch.zeros(layer_num, self.cache_size + 1).cuda()

#         self.ratio = ratio
#         self.protect_size = 1
#         self.flag = True
#         self.distance = distance
#         self.layer_num = layer_num

#         self.attn_weight = attn_weight
#         self.cross_modal_weight = cross_modal_weight

#         # NEW: modality boundary
#         self.text_token_count = text_token_count

#         self.selected_idx = 0

#     def _build_modality_masks(self, seq_len, device):
#         """
#         NEW:
#         Build boolean masks for text vs vision tokens.

#         Assumption:
#         [0 ... text_token_count-1] = text
#         [text_token_count ... seq_len-1] = vision
#         """
#         assert self.text_token_count is not None, "text_token_count must be set for multimodal scoring"

#         text_cnt = min(self.text_token_count, seq_len)
#         text_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
#         text_mask[:text_cnt] = True

#         vision_mask = ~text_mask
#         return text_mask, vision_mask

#     def __call__(self, past_key_values, num_of_token=None, attentions=None):
#         if past_key_values is None:
#             return None

#         attn_score = [attention for attention in attentions]
#         seq_len = past_key_values[0][0].size(self.k_seq_dim)

#         # Concatenate layers, average over heads
#         attn_score = torch.cat(attn_score, dim=0)                 # [layer, heads, q, k] assumed after cat
#         attn_score = attn_score.mean(dim=1, keepdim=False)        # [layer, q, k]

#         # NEW: modality masks
#         text_mask, vision_mask = self._build_modality_masks(seq_len, attn_score.device)

#         # Update importance scores only during first full scoring pass
#         if attn_score.shape[-2] > 1:
#             assert self.flag is True

#             for idx in range(attn_score.shape[-1]):
#                 # Attention of query token idx to all valid previous tokens [0..idx]
#                 cur_score = attn_score[:, idx, :idx + 1]          # [layer, idx+1]

#                 # -------------------------
#                 # ORIGINAL METRIC
#                 # -------------------------
#                 self.score_sum[:, :cur_score.shape[-1]] += cur_score

#                 # -------------------------
#                 # NEW: CROSS-MODAL SCORE
#                 # -------------------------
#                 cur_text_mask = text_mask[:idx + 1]               # [idx+1]
#                 cur_vision_mask = vision_mask[:idx + 1]           # [idx+1]

#                 # Query token modality
#                 is_query_text = text_mask[idx].item()

#                 if is_query_text:
#                     # For a text query, only its attention to vision keys contributes
#                     cross_score = cur_score * cur_vision_mask.unsqueeze(0).float()
#                 else:
#                     # For a vision query, only its attention to text keys contributes
#                     cross_score = cur_score * cur_text_mask.unsqueeze(0).float()

#                 self.cross_modal_score_sum[:, :cur_score.shape[-1]] += cross_score

#         forget_num = int(seq_len - num_of_token * (1 - self.ratio))
#         if forget_num <= 0:
#             return past_key_values

#         if forget_num > 1:
#             assert self.flag is True
#             self.flag = False

#             selected_idx_all = []
#             merge_idx_all = []
#             throw_idx_all = []

#             for idx in range(self.layer_num):
#                 start = self.start_size
#                 end = seq_len - self.protect_size

#                 # Candidate range
#                 raw_score = self.score_sum[idx, start:end]
#                 cross_score = self.cross_modal_score_sum[idx, start:end]

#                 local_text_mask = text_mask[start:end]
#                 local_vision_mask = vision_mask[start:end]

#                 # -------------------------
#                 # NEW: modality-aware normalization
#                 # Normalize text tokens only against text tokens,
#                 # and vision tokens only against vision tokens.
#                 # -------------------------
#                 raw_score_norm = torch.zeros_like(raw_score)
#                 cross_score_norm = torch.zeros_like(cross_score)

#                 if local_text_mask.any():
#                     raw_text = raw_score[local_text_mask]
#                     cross_text = cross_score[local_text_mask]

#                     raw_score_norm[local_text_mask] = raw_text / (raw_text.max() + 1e-6)
#                     cross_score_norm[local_text_mask] = cross_text / (cross_text.max() + 1e-6)

#                 if local_vision_mask.any():
#                     raw_vision = raw_score[local_vision_mask]
#                     cross_vision = cross_score[local_vision_mask]

#                     raw_score_norm[local_vision_mask] = raw_vision / (raw_vision.max() + 1e-6)
#                     cross_score_norm[local_vision_mask] = cross_vision / (cross_vision.max() + 1e-6)

#                 # -------------------------
#                 # NEW: combined multimodal importance
#                 # -------------------------
#                 combined_score = (
#                     self.attn_weight * raw_score_norm +
#                     self.cross_modal_weight * cross_score_norm
#                 )

#                 # Remove lowest-scoring tokens, keep highest-scoring tokens
#                 order = torch.argsort(combined_score)   # low -> high
#                 throw_idx = order[:forget_num] + start
#                 selected_idx = order[forget_num:] + start

#                 # Merge each removed token into nearest kept token
#                 merge_idx = []
#                 for i in range(len(throw_idx)):
#                     nearest_idx = torch.abs(selected_idx - throw_idx[i]).argmin()
#                     merge_idx.append(selected_idx[nearest_idx].unsqueeze(0))
#                 merge_idx = torch.cat(merge_idx)

#                 # Always keep prefix and last token
#                 selected_idx = torch.cat(
#                     [
#                         torch.arange(self.start_size, device=raw_score.device),
#                         selected_idx,
#                         torch.tensor([seq_len - self.protect_size], device=raw_score.device),
#                     ],
#                     dim=0
#                 )

#                 selected_idx_all.append(selected_idx)
#                 merge_idx_all.append(merge_idx)
#                 throw_idx_all.append(throw_idx)

#             if self.distance > 0:
#                 self.selected_idx = self.distance
#             else:
#                 self.selected_idx = seq_len - forget_num + self.distance

#             past_key_values_return = []
#             for idx, (k, v) in enumerate(past_key_values):
#                 selected_idx = selected_idx_all[idx]
#                 merge_idx = merge_idx_all[idx]
#                 throw_idx = throw_idx_all[idx]

#                 k_forget = k.gather(
#                     dim=-2,
#                     index=throw_idx.view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, k.shape[-1])
#                 )
#                 v_forget = v.gather(
#                     dim=-2,
#                     index=throw_idx.view(1, 1, -1, 1).expand(v.shape[0], v.shape[1], -1, v.shape[-1])
#                 )

#                 k = k.scatter_reduce(
#                     -2,
#                     merge_idx.view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, k.shape[-1]),
#                     k_forget,
#                     'mean'
#                 )
#                 v = v.scatter_reduce(
#                     -2,
#                     merge_idx.view(1, 1, -1, 1).expand(v.shape[0], v.shape[1], -1, v.shape[-1]),
#                     v_forget,
#                     'mean'
#                 )

#                 k_new = k.gather(
#                     dim=-2,
#                     index=selected_idx.view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, k.shape[-1])
#                 )
#                 v_new = v.gather(
#                     dim=-2,
#                     index=selected_idx.view(1, 1, -1, 1).expand(v.shape[0], v.shape[1], -1, v.shape[-1])
#                 )

#                 past_key_values_return.append([k_new, v_new])

#             return past_key_values_return

#         else:
#             selected_idx = self.selected_idx
#             return [[
#                 torch.cat(
#                     [
#                         self.k_slice(k, 0, selected_idx),
#                         self.k_slice(k, selected_idx + 1, seq_len),
#                     ],
#                     dim=self.k_seq_dim,
#                 ),
#                 torch.cat(
#                     [
#                         self.v_slice(v, 0, selected_idx),
#                         self.v_slice(v, selected_idx + 1, seq_len),
#                     ],
#                     dim=self.v_seq_dim,
#                 )
#             ] for k, v in past_key_values]

# class ElasticCache:
#     def __init__(
#         self,
#         start_size=4,          # always keep first few tokens (prefix)
#         recent_size=512,       # target size for recent tokens
#         k_seq_dim=2,           # sequence dimension for K
#         v_seq_dim=2,           # sequence dimension for V
#         ratio=0.,              # controls how aggressively to forget
#         distance=-25,          # used later for incremental removal
#         layer_num=40,          # number of transformer layers
#         attn_weight=1.0,       # NEW: weight for attention importance
#         relation_weight=1.0,   # NEW: weight for connectivity importance
#     ):
#         self.start_size = start_size
#         self.recent_size = recent_size
#         self.cache_size = start_size + recent_size  # total target cache size

#         self.k_seq_dim = k_seq_dim
#         self.v_seq_dim = v_seq_dim

#         # helper slicing functions for removing tokens later
#         self.k_slice = DIM_TO_SLICE[k_seq_dim]
#         self.v_slice = DIM_TO_SLICE[v_seq_dim]

#         # accumulated attention importance per layer per token
#         self.score_sum = torch.zeros(layer_num, self.cache_size + 1).cuda()

#         # NEW: accumulated connectivity (how many above-mean relations per token)
#         self.relation_count_sum = torch.zeros(layer_num, self.cache_size + 1).cuda()

#         self.ratio = ratio
#         self.protect_size = 1      # always keep last token
#         self.flag = True           # ensures main compression runs only once
#         self.distance = distance
#         self.layer_num = layer_num

#         # NEW: weights to combine attention + connectivity
#         self.attn_weight = attn_weight
#         self.relation_weight = relation_weight

#         self.selected_idx = 0      # used for later incremental deletion

#     def __call__(self, past_key_values, num_of_token=None, attentions=None):
#         # if no KV cache, nothing to process
#         if past_key_values is None:
#             return None

#         # collect attention tensors from all layers
#         attn_score = [attention for attention in attentions]

#         # current sequence length
#         seq_len = past_key_values[0][0].size(self.k_seq_dim)

#         # stack attention across layers
#         attn_score = torch.cat(attn_score, dim=0)

#         # average across heads → [layer, query_pos, key_pos]
#         attn_score = attn_score.mean(dim=1, keepdim=False)

#         # update importance scores (only first time)
#         if attn_score.shape[-2] > 1:
#             assert self.flag is True

#             # iterate over query positions
#             for idx in range(attn_score.shape[-1]):

#                 # attention from current token to all previous tokens
#                 cur_score = attn_score[:, idx, :idx + 1]

#                 # -------------------------
#                 # ORIGINAL METRIC
#                 # -------------------------
#                 # accumulate total attention received by each token
#                 self.score_sum[:, :cur_score.shape[-1]] += cur_score

#                 # -------------------------
#                 # NEW METRIC: CONNECTIVITY
#                 # -------------------------

#                 # compute mean attention for each token vector
#                 cur_mean = cur_score.mean(dim=-1, keepdim=True)   # NEW

#                 # mark entries above mean → strong relations
#                 relation_mask = (cur_score > cur_mean).float()    # NEW

#                 # accumulate how many strong relations each token has
#                 self.relation_count_sum[:, :cur_score.shape[-1]] += relation_mask  # NEW

#         # compute how many tokens to forget
#         forget_num = int(seq_len - num_of_token * (1 - self.ratio))

#         if forget_num <= 0:
#             # no compression needed
#             return past_key_values

#         else:
#             # -------------------------
#             # MAIN COMPRESSION STEP
#             # -------------------------
#             if forget_num > 1:
#                 assert self.flag is True
#                 self.flag = False   # ensure this block runs only once

#                 selected_idx_all = []  # tokens to keep
#                 merge_idx_all = []     # where to merge removed tokens
#                 throw_idx_all = []     # tokens to remove

#                 for idx in range(self.layer_num):

#                     start = self.start_size
#                     end = seq_len - self.protect_size

#                     # ORIGINAL attention score
#                     raw_score = self.score_sum[idx, start:end]

#                     # NEW connectivity score
#                     rel_score = self.relation_count_sum[idx, start:end]

#                     # -------------------------
#                     # NORMALIZATION
#                     # -------------------------
#                     # prevent scale mismatch between metrics
#                     raw_score_norm = raw_score / (raw_score.max() + 1e-6)   # NEW
#                     rel_score_norm = rel_score / (rel_score.max() + 1e-6)   # NEW

#                     # -------------------------
#                     # COMBINED IMPORTANCE
#                     # -------------------------
#                     combined_score = (
#                         self.attn_weight * raw_score_norm +
#                         self.relation_weight * rel_score_norm
#                     )   # NEW

#                     # sort tokens by importance (low → high)
#                     order = torch.argsort(combined_score)

#                     # lowest scores → removed
#                     throw_idx = order[:forget_num] + start

#                     # highest scores → kept
#                     selected_idx = order[forget_num:] + start

#                     # -------------------------
#                     # MERGE LOGIC
#                     # -------------------------
#                     # map each removed token to nearest kept token
#                     merge_idx = []
#                     for i in range(len(throw_idx)):
#                         nearest_idx = torch.abs(selected_idx - throw_idx[i]).argmin()
#                         merge_idx.append(selected_idx[nearest_idx].unsqueeze(0))
#                     merge_idx = torch.cat(merge_idx)

#                     # always keep:
#                     # - prefix tokens
#                     # - selected tokens
#                     # - last token
#                     selected_idx = torch.cat(
#                         [
#                             torch.arange(self.start_size).cuda(),
#                             selected_idx,
#                             torch.tensor([seq_len - self.protect_size]).cuda()
#                         ],
#                         dim=0
#                     )

#                     selected_idx_all.append(selected_idx)
#                     merge_idx_all.append(merge_idx)
#                     throw_idx_all.append(throw_idx)

#                 # choose index for later incremental deletion
#                 if self.distance > 0:
#                     self.selected_idx = self.distance
#                 else:
#                     self.selected_idx = seq_len - forget_num + self.distance

#                 # -------------------------
#                 # APPLY COMPRESSION
#                 # -------------------------
#                 past_key_values_return = []

#                 for idx, (k, v) in enumerate(past_key_values):

#                     selected_idx = selected_idx_all[idx]
#                     merge_idx = merge_idx_all[idx]
#                     throw_idx = throw_idx_all[idx]

#                     # gather removed tokens
#                     k_forget = k.gather(
#                         dim=-2,
#                         index=throw_idx.view(1,1,-1,1).expand(
#                             k.shape[0], k.shape[1], -1, k.shape[-1])
#                     )
#                     v_forget = v.gather(
#                         dim=-2,
#                         index=throw_idx.view(1,1,-1,1).expand(
#                             v.shape[0], v.shape[1], -1, v.shape[-1])
#                     )

#                     # merge removed tokens into nearest kept tokens (mean)
#                     k = k.scatter_reduce(
#                         -2,
#                         merge_idx.view(1,1,-1,1).expand(
#                             k.shape[0], k.shape[1], -1, k.shape[-1]),
#                         k_forget,
#                         'mean'
#                     )
#                     v = v.scatter_reduce(
#                         -2,
#                         merge_idx.view(1,1,-1,1).expand(
#                             v.shape[0], v.shape[1], -1, v.shape[-1]),
#                         v_forget,
#                         'mean'
#                     )

#                     # gather only selected tokens → compressed cache
#                     k_new = k.gather(
#                         dim=-2,
#                         index=selected_idx.view(1,1,-1,1).expand(
#                             k.shape[0], k.shape[1], -1, k.shape[-1])
#                     )
#                     v_new = v.gather(
#                         dim=-2,
#                         index=selected_idx.view(1,1,-1,1).expand(
#                             v.shape[0], v.shape[1], -1, v.shape[-1])
#                     )

#                     past_key_values_return.append([k_new, v_new])

#                 return past_key_values_return

#             # -------------------------
#             # INCREMENTAL STEP (remove 1 token)
#             # -------------------------
#             else:
#                 selected_idx = self.selected_idx

#                 return [[
#                     torch.cat(
#                         [
#                             self.k_slice(k, 0, selected_idx),
#                             self.k_slice(k, selected_idx + 1, seq_len),
#                         ],
#                         dim=self.k_seq_dim,
#                     ),
#                     torch.cat(
#                         [
#                             self.v_slice(v, 0, selected_idx),
#                             self.v_slice(v, selected_idx + 1, seq_len),
#                         ],
#                         dim=self.v_seq_dim,
#                     )
#                 ] for k, v in past_key_values]

class ElasticCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.,
        distance=-25,
        layer_num=40,
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

        self.score_sum = torch.zeros(layer_num, self.cache_size + 1).cuda()
        self.ratio = ratio
        self.protect_size = 1
        self.flag = True
        self.distance = distance
        self.layer_num = layer_num

        self.selected_idx = 0

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        attn_score = [attention for attention in attentions]
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        # update attn score
        attn_score = torch.cat(attn_score, dim=0)
        attn_score = attn_score.mean(dim=1, keepdim=False)
        if attn_score.shape[-2] > 1:
            #assert self.flag is True # only use for the first time
            if self.flag is not True: #added by tabassum
                return past_key_values #added by tabassum
            for idx in range(attn_score.shape[-1]):
                cur_score = attn_score[:, idx, :idx+1]
                self.score_sum[:, :(cur_score.shape[-1])] += cur_score
        else:
            pass

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        if forget_num <= 0:
            return past_key_values
        else:
            if forget_num > 1:
                #assert self.flag is True
                if self.flag is not True: #added by tabassum
                    return past_key_values #added by tabassum
                self.flag = False
                self.flag = False

                selected_idx_all = []
                merge_idx_all = []
                throw_idx_all = []
                for idx in range(self.layer_num):
                    selected_idx = torch.where(torch.argsort(self.score_sum[idx, self.start_size:(seq_len - self.protect_size)]) > forget_num)[0] + self.start_size
                    throw_idx = torch.where(torch.argsort(self.score_sum[idx, self.start_size:(seq_len - self.protect_size)]) <= forget_num)[0]
                    merge_idx = []
                    for i in range(len(throw_idx)):
                        merge_idx.append(selected_idx[torch.abs((selected_idx - throw_idx[i])).argmin()].unsqueeze(0))
                    merge_idx = torch.cat(merge_idx)

                    selected_idx = torch.cat([torch.arange(self.start_size).cuda(), selected_idx, torch.tensor([seq_len - self.protect_size]).cuda()], dim=0) # the last token is always kept

                    selected_idx_all.append(selected_idx)
                    merge_idx_all.append(merge_idx)
                    throw_idx_all.append(throw_idx)

                if self.distance > 0:
                    self.selected_idx = self.distance
                else:
                    self.selected_idx = seq_len - forget_num + self.distance

                past_key_values_return = []
                for idx, (k, v) in enumerate(past_key_values):
                    selected_idx = selected_idx_all[idx]
                    merge_idx = merge_idx_all[idx]
                    throw_idx = throw_idx_all[idx]

                    k_forget = k.gather(dim=-2, index=throw_idx.view(1,1,-1,1).expand(k.shape[0], k.shape[1], -1 ,k.shape[-1]))
                    v_forget = v.gather(dim=-2, index=throw_idx.view(1,1,-1,1).expand(v.shape[0], v.shape[1], -1 ,v.shape[-1]))

                    k = k.scatter_reduce(-2, merge_idx.view(1,1,-1,1).expand(k.shape[0], k.shape[1], -1 ,k.shape[-1]), k_forget, 'mean')
                    v = v.scatter_reduce(-2, merge_idx.view(1,1,-1,1).expand(v.shape[0], v.shape[1], -1 ,v.shape[-1]), v_forget, 'mean')

                    k_new = k.gather(dim=-2, index=selected_idx.view(1,1,-1,1).expand(k.shape[0], k.shape[1], -1 ,k.shape[-1]))
                    v_new = v.gather(dim=-2, index=selected_idx.view(1,1,-1,1).expand(v.shape[0], v.shape[1], -1 ,v.shape[-1]))

                    past_key_values_return.append([k_new, v_new])
                return past_key_values_return
            else:
                selected_idx = self.selected_idx
                return [[torch.cat([self.k_slice(k, 0, selected_idx), self.k_slice(k, (selected_idx+1), seq_len),],
                            dim=self.k_seq_dim,),
                        torch.cat([self.v_slice(v, 0, selected_idx), self.v_slice(v, (selected_idx+1), seq_len),],
                            dim=self.v_seq_dim,)]
                    for k, v in past_key_values]
            

class LocalCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.ratio = ratio

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        if forget_num <= 0:
            return past_key_values
        else:
            return [[torch.cat([self.k_slice(k, 0, self.start_size), self.k_slice(k, forget_num + self.start_size, seq_len),],
                        dim=self.k_seq_dim,),
                    torch.cat([self.v_slice(v, 0, self.start_size), self.v_slice(v, forget_num + self.start_size, seq_len),],
                        dim=self.v_seq_dim,),]
                for k, v in past_key_values]
        

class H2OCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

        self.score_sum = torch.zeros(self.cache_size + 1).cuda()
        self.ratio = ratio
        self.protect_size = 1
        self.flag = True

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        attn_score = [attention for attention in attentions]
        past_key_values_new = tuple(x for x in past_key_values)
        seq_len = past_key_values_new[0][0].size(self.k_seq_dim)
        
        # update attn score
        attn_score = torch.cat(attn_score, dim=0)
        attn_score = attn_score.mean(dim=1, keepdim=False).mean(dim=0, keepdim=False)

        if attn_score.shape[-2] > 1:
            assert self.flag is True # only use for the first time
            for idx in range(attn_score.shape[-1]):
                cur_score = attn_score[idx][:idx+1]
                self.score_sum[:len(cur_score)] += cur_score
        else:
            attn_score = attn_score.squeeze(0)
            self.score_sum[:seq_len] += attn_score

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        self.protect_size = 1
        if forget_num <= 0:
            return past_key_values_new
        else:
            if forget_num > 1:
                assert self.flag is True
                self.flag = False
                selected_idx = torch.where(torch.argsort(self.score_sum[:(seq_len - self.protect_size)]) > forget_num)[0]
                selected_idx = torch.cat([selected_idx, torch.arange(seq_len - self.protect_size, seq_len).cuda()], dim=0)
                past_key_values_return = []
                for k, v in past_key_values_new:
                    k_new = k.gather(dim=-2, index=selected_idx.view(1,1,-1,1).expand(k.shape[0], k.shape[1], -1 ,k.shape[-1]))
                    v_new = v.gather(dim=-2, index=selected_idx.view(1,1,-1,1).expand(v.shape[0], v.shape[1], -1 ,v.shape[-1]))
                    past_key_values_return.append([k_new, v_new])
                
                return past_key_values_return
            else:
                selected_idx = self.score_sum[self.start_size:(seq_len - self.protect_size)].argmin() + self.start_size
                self.score_sum[(selected_idx):-1] = self.score_sum[(selected_idx+1):].clone()
                
                return [[torch.cat([self.k_slice(k, 0, selected_idx), self.k_slice(k, (selected_idx+1), seq_len),],
                            dim=self.k_seq_dim,),
                        torch.cat([self.v_slice(v, 0, selected_idx), self.v_slice(v, (selected_idx+1), seq_len),],
                            dim=self.v_seq_dim,)]
                    for k, v in past_key_values_new]