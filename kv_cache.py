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

# bad result, don't use it.
# class MultimodalElasticCache:
#     def __init__(
#         self,
#         start_size=4,
#         recent_size=512,
#         k_seq_dim=2,
#         v_seq_dim=2,
#         ratio=0.0,
#         distance=-25,
#         layer_num=40,
#         attn_weight=1.0,
#         text_image_weight=0.8,
#         image_text_weight=0.8,
#         image_start=None,
#         image_token_len=576,
#     ):
#         self.start_size = start_size
#         self.recent_size = recent_size
#         self.cache_size = start_size + recent_size
#         self.k_seq_dim = k_seq_dim
#         self.v_seq_dim = v_seq_dim
#         self.k_slice = DIM_TO_SLICE[k_seq_dim]
#         self.v_slice = DIM_TO_SLICE[v_seq_dim]

#         self.score_sum = torch.zeros(layer_num, self.cache_size + 1).cuda()
#         self.text_image_score_sum = torch.zeros(layer_num, self.cache_size + 1).cuda()
#         self.image_text_score_sum = torch.zeros(layer_num, self.cache_size + 1).cuda()

#         self.ratio = ratio
#         self.protect_size = 1
#         self.flag = True
#         self.distance = distance
#         self.layer_num = layer_num

#         self.attn_weight = attn_weight
#         self.text_image_weight = text_image_weight
#         self.image_text_weight = image_text_weight

#         self.image_start = image_start
#         self.image_token_len = image_token_len

#         self.selected_idx = 0

#     def _build_modality_masks(self, seq_len, device):
#         text_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
#         image_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

#         if self.image_start is None:
#             return text_mask, image_mask

#         image_end = min(self.image_start + self.image_token_len, seq_len)

#         if self.image_start < seq_len:
#             text_mask[self.image_start:image_end] = False
#             image_mask[self.image_start:image_end] = True

#         return text_mask, image_mask

#     def reset(self):
#         self.score_sum.zero_()
#         self.text_image_score_sum.zero_()
#         self.image_text_score_sum.zero_()
#         self.flag = True
#         self.selected_idx = 0

#     def _nearest_same_modality(self, throw_idx, selected_idx, text_mask, image_mask):
#         merge_idx = []

#         for t in throw_idx:
#             if image_mask[t].item():
#                 same_selected = selected_idx[image_mask[selected_idx]]
#             else:
#                 same_selected = selected_idx[text_mask[selected_idx]]

#             if same_selected.numel() == 0:
#                 same_selected = selected_idx

#             nearest_idx = torch.abs(same_selected - t).argmin()
#             merge_idx.append(same_selected[nearest_idx].unsqueeze(0))

#         return torch.cat(merge_idx)

#     def __call__(self, past_key_values, num_of_token=None, attentions=None):
#         if past_key_values is None:
#             return None

#         seq_len = past_key_values[0][0].size(self.k_seq_dim)

#         attn_score = [attention for attention in attentions]
#         attn_score = torch.cat(attn_score, dim=0)
#         attn_score = attn_score.mean(dim=1, keepdim=False)

#         text_mask, image_mask = self._build_modality_masks(seq_len, attn_score.device)

#         if attn_score.shape[-2] > 1:
#             if self.flag is not True:
#                 return past_key_values

#             for idx in range(attn_score.shape[-2]):
#                 cur_score = attn_score[:, idx, :idx + 1]

#                 self.score_sum[:, :cur_score.shape[-1]] += cur_score

#                 if text_mask[idx].item():
#                     text_image_score = cur_score * image_mask[:idx + 1].unsqueeze(0).float()
#                     self.text_image_score_sum[:, :cur_score.shape[-1]] += text_image_score
#                 else:
#                     image_text_score = cur_score * text_mask[:idx + 1].unsqueeze(0).float()
#                     self.image_text_score_sum[:, :cur_score.shape[-1]] += image_text_score

#         forget_num = int(seq_len - num_of_token * (1 - self.ratio))

#         if forget_num <= 0:
#             return past_key_values

#         if forget_num > 1:
#             if self.flag is not True:
#                 return past_key_values

#             self.flag = False

#             selected_idx_all = []
#             merge_idx_all = []
#             throw_idx_all = []

#             for layer_idx in range(self.layer_num):
#                 start = self.start_size
#                 end = seq_len - self.protect_size

#                 raw_score = self.score_sum[layer_idx, start:end]
#                 text_image_score = self.text_image_score_sum[layer_idx, start:end]
#                 image_text_score = self.image_text_score_sum[layer_idx, start:end]

#                 local_text_mask = text_mask[start:end]
#                 local_image_mask = image_mask[start:end]

#                 raw_score_norm = raw_score / (raw_score.max() + 1e-6)

#                 text_image_norm = torch.zeros_like(text_image_score)
#                 if local_image_mask.any():
#                     vals = text_image_score[local_image_mask]
#                     text_image_norm[local_image_mask] = vals / (vals.max() + 1e-6)

#                 image_text_norm = torch.zeros_like(image_text_score)
#                 if local_text_mask.any():
#                     vals = image_text_score[local_text_mask]
#                     image_text_norm[local_text_mask] = vals / (vals.max() + 1e-6)

#                 combined_score = self.attn_weight * raw_score_norm

#                 combined_score[local_image_mask] += (
#                     self.text_image_weight * text_image_norm[local_image_mask]
#                 )

#                 combined_score[local_text_mask] += (
#                     self.image_text_weight * image_text_norm[local_text_mask]
#                 )

#                 order = torch.argsort(combined_score)

#                 throw_idx = order[:forget_num] + start
#                 selected_idx = order[forget_num:] + start

#                 merge_idx = self._nearest_same_modality(
#                     throw_idx,
#                     selected_idx,
#                     text_mask,
#                     image_mask,
#                 )

#                 selected_idx = torch.cat(
#                     [
#                         torch.arange(self.start_size, device=combined_score.device),
#                         selected_idx,
#                         torch.tensor([seq_len - self.protect_size], device=combined_score.device),
#                     ],
#                     dim=0,
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
#                 selected_idx = selected_idx_all[layer_idx].to(k.device)
#                 merge_idx = merge_idx_all[layer_idx].to(k.device)
#                 throw_idx = throw_idx_all[layer_idx].to(k.device)

#                 k_forget = k.gather(
#                     dim=-2,
#                     index=throw_idx.view(1, 1, -1, 1).expand(
#                         k.shape[0], k.shape[1], -1, k.shape[-1]
#                     ),
#                 )

#                 v_forget = v.gather(
#                     dim=-2,
#                     index=throw_idx.view(1, 1, -1, 1).expand(
#                         v.shape[0], v.shape[1], -1, v.shape[-1]
#                     ),
#                 )

#                 k = k.scatter_reduce(
#                     -2,
#                     merge_idx.view(1, 1, -1, 1).expand(
#                         k.shape[0], k.shape[1], -1, k.shape[-1]
#                     ),
#                     k_forget,
#                     "mean",
#                 )

#                 v = v.scatter_reduce(
#                     -2,
#                     merge_idx.view(1, 1, -1, 1).expand(
#                         v.shape[0], v.shape[1], -1, v.shape[-1]
#                     ),
#                     v_forget,
#                     "mean",
#                 )

#                 k_new = k.gather(
#                     dim=-2,
#                     index=selected_idx.view(1, 1, -1, 1).expand(
#                         k.shape[0], k.shape[1], -1, k.shape[-1]
#                     ),
#                 )

#                 v_new = v.gather(
#                     dim=-2,
#                     index=selected_idx.view(1, 1, -1, 1).expand(
#                         v.shape[0], v.shape[1], -1, v.shape[-1]
#                     ),
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
#                 ),
#             ] for k, v in past_key_values]

#applying connection metric only for text tokens but taking text-text approach

class MultimodalElasticCache:
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
        # === MODALITY-SEGREGATED MERGING (new) ===
        # grid_size: side length of the square patch grid an image is
        # flattened into (24 for LLaVA-1.5's 576 = 24x24 image tokens).
        # Used only for image-token merge-target distance below.
        grid_size=24,
        # segregate_merge=True (new default): restrict merge targets to
        # same-modality anchors, using 2D grid distance for image tokens
        # and the original raw index distance for text tokens. Set to
        # False to fall back to the exact original unconstrained-merge
        # behavior (useful for a clean side-by-side ablation with the
        # same image_start/image_token_len configuration).
        segregate_merge=True,
        # === END MODALITY-SEGREGATED MERGING ===
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

        # === MODALITY-SEGREGATED MERGING (new) ===
        self.grid_size = grid_size
        self.segregate_merge = segregate_merge
        # === END MODALITY-SEGREGATED MERGING ===

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

    # === MODALITY-SEGREGATED MERGING (new) ===
    def _nearest_same_modality_grid(self, throw_idx, selected_idx, text_mask, image_mask):
        """Restrict each thrown token's merge target to a same-modality
        anchor. Text tokens: raw sequence-index distance, IDENTICAL formula
        to the original unconstrained version below -- so text-token merge
        behavior only changes because the candidate POOL is now restricted
        to text anchors, not because the distance metric changed. Image
        tokens: 2D spatial-grid distance instead of index distance, since
        LLaVA's grid_size x grid_size patch grid is flattened into a 1D
        sequence in raster order -- two patches that are vertically adjacent
        in the actual image can be `grid_size` positions apart in the
        sequence, so raw index distance systematically favors horizontal
        neighbors and misrepresents true spatial adjacency.

        NOTE ON SCALE: unlike the plain (non-multimodal) ElasticCache class,
        both throw_idx and selected_idx here are ALREADY absolute sequence
        positions (both computed as `order[...] + start` above) -- so no
        scale correction is needed, unlike the standalone patch."""
        merge_idx = []
        image_selected_mask = image_mask[selected_idx]
        text_selected_mask = text_mask[selected_idx]

        for i in range(len(throw_idx)):
            t = throw_idx[i]
            t_is_image = image_mask[t].item()
            same_modality = selected_idx[image_selected_mask] if t_is_image else selected_idx[text_selected_mask]
            if same_modality.numel() == 0:
                # fallback: no same-modality anchor survived at this layer, use full pool
                same_modality = selected_idx

            if t_is_image and same_modality.numel() > 0 and image_selected_mask.any():
                t_local = (t - self.image_start).float()
                cand_local = (same_modality - self.image_start).float()
                t_row, t_col = torch.div(t_local, self.grid_size, rounding_mode='floor'), t_local % self.grid_size
                c_row, c_col = torch.div(cand_local, self.grid_size, rounding_mode='floor'), cand_local % self.grid_size
                dist = torch.sqrt((t_row - c_row) ** 2 + (t_col - c_col) ** 2)
                nearest = same_modality[dist.argmin()]
            else:
                # text tokens (or image fallback with no image candidates):
                # raw index distance, same formula as the original code.
                nearest = same_modality[torch.abs(same_modality - t).argmin()]
            merge_idx.append(nearest.unsqueeze(0))
        return torch.cat(merge_idx)
    # === END MODALITY-SEGREGATED MERGING ===

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

                # === MODALITY-SEGREGATED MERGING (new) ===
                if self.segregate_merge:
                    merge_idx = self._nearest_same_modality_grid(throw_idx, selected_idx, text_mask, image_mask)
                else:
                    # original unconstrained behavior, unchanged
                    merge_idx = []
                    for i in range(len(throw_idx)):
                        nearest_idx = torch.abs(selected_idx - throw_idx[i]).argmin()
                        merge_idx.append(selected_idx[nearest_idx].unsqueeze(0))
                    merge_idx = torch.cat(merge_idx)
                # === END MODALITY-SEGREGATED MERGING ===

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


class MultimodalHelasticCache:
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
        #conn_weight=0.8,
        conn_weight=1.0,
        image_start=None,
        image_token_len=576,
        # === MODALITY-SEGREGATED MERGING (new) ===
        # grid_size: side length of the square patch grid an image is
        # flattened into (24 for LLaVA-1.5's 576 = 24x24 image tokens).
        # Used only for image-token merge-target distance below.
        grid_size=24,
        # segregate_merge=True (new default): restrict merge targets to
        # same-modality anchors, using 2D grid distance for image tokens
        # and the original raw index distance for text tokens. Set to
        # False to fall back to the exact original unconstrained-merge
        # behavior (useful for a clean side-by-side ablation with the
        # same image_start/image_token_len configuration).
        segregate_merge=True,
        # === END MODALITY-SEGREGATED MERGING ===
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

        # === MODALITY-SEGREGATED MERGING (new) ===
        self.grid_size = grid_size
        self.segregate_merge = segregate_merge
        # === END MODALITY-SEGREGATED MERGING ===

        self.selected_idx = 0

        # === SINGLE-TURN-SAFE MODALITY SEGREGATION (new) ===
        # This class's merge branch can fire more than once per session
        # (its `assert self.flag is True` guard never actually blocks a
        # second round -- the line that would set self.flag = False is
        # commented out above). But _build_modality_masks() assumes
        # image_start/image_token_len still describe the CURRENT cache
        # layout, which is only true before any merging has happened.
        # After the first merge round, positions have shifted and the
        # mask would silently misclassify tokens. Until per-slot modality
        # tracking is implemented, segregation is only safe on the FIRST
        # merge round; every round after that falls back to the original
        # unconstrained merge automatically, regardless of segregate_merge.
        self._first_merge_done = False
        # === END SINGLE-TURN-SAFE MODALITY SEGREGATION ===

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

    # === MODALITY-SEGREGATED MERGING (new) ===
    def _nearest_same_modality_grid(self, throw_idx, selected_idx, text_mask, image_mask):
        """Restrict each thrown token's merge target to a same-modality
        anchor. Text tokens: raw sequence-index distance, IDENTICAL formula
        to the original unconstrained version below -- so text-token merge
        behavior only changes because the candidate POOL is now restricted
        to text anchors, not because the distance metric changed. Image
        tokens: 2D spatial-grid distance instead of index distance, since
        LLaVA's grid_size x grid_size patch grid is flattened into a 1D
        sequence in raster order -- two patches that are vertically adjacent
        in the actual image can be `grid_size` positions apart in the
        sequence, so raw index distance systematically favors horizontal
        neighbors and misrepresents true spatial adjacency.

        NOTE ON SCALE: both throw_idx and selected_idx here are ALREADY
        absolute sequence positions (both computed as `... + self.start_size`
        below) -- same convention as MultimodalElasticCache, so no scale
        correction is needed."""
        merge_idx = []
        image_selected_mask = image_mask[selected_idx]
        text_selected_mask = text_mask[selected_idx]

        for i in range(len(throw_idx)):
            t = throw_idx[i]
            t_is_image = image_mask[t].item()
            same_modality = selected_idx[image_selected_mask] if t_is_image else selected_idx[text_selected_mask]
            if same_modality.numel() == 0:
                # fallback: no same-modality anchor survived at this layer, use full pool
                same_modality = selected_idx

            if t_is_image and same_modality.numel() > 0 and image_selected_mask.any():
                t_local = (t - self.image_start).float()
                cand_local = (same_modality - self.image_start).float()
                t_row, t_col = torch.div(t_local, self.grid_size, rounding_mode='floor'), t_local % self.grid_size
                c_row, c_col = torch.div(cand_local, self.grid_size, rounding_mode='floor'), cand_local % self.grid_size
                dist = torch.sqrt((t_row - c_row) ** 2 + (t_col - c_col) ** 2)
                nearest = same_modality[dist.argmin()]
            else:
                # text tokens (or image fallback with no image candidates):
                # raw index distance, same formula as the original code.
                nearest = same_modality[torch.abs(same_modality - t).argmin()]
            merge_idx.append(nearest.unsqueeze(0))
        return torch.cat(merge_idx)
    # === END MODALITY-SEGREGATED MERGING ===

    def reset(self):
        self.score_sum.zero_()
        self.conn_score_sum.zero_()
        self.flag = True
        self.selected_idx = 0
        # === SINGLE-TURN-SAFE MODALITY SEGREGATION (new) ===
        self._first_merge_done = False
        # === END SINGLE-TURN-SAFE MODALITY SEGREGATION ===

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None

        attn_score = [attention for attention in attentions]
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        # # update attn score
        # 1. Update Attention Score (H2O with Decay)
        attn_score = torch.cat(attn_score, dim=0)
        attn_score = attn_score.mean(dim=1, keepdim=False)

        # [Fix] Expand score_sum if the current sequence is longer than the buffer
        if seq_len > self.score_sum.shape[-1]:
            new_score_sum = torch.zeros(self.layer_num, seq_len + 1).to(self.score_sum.device)
            new_score_sum[:, :self.score_sum.shape[-1]] = self.score_sum
            self.score_sum = new_score_sum

        if seq_len > self.conn_score_sum.shape[-1]:
            new_conn_score_sum = torch.zeros(self.layer_num, seq_len + 1, device=self.conn_score_sum.device)
            new_conn_score_sum[:, :self.conn_score_sum.shape[-1]] = self.conn_score_sum
            self.conn_score_sum = new_conn_score_sum
        
        text_mask, image_mask = self._build_modality_masks(seq_len, attn_score.device)

        if attn_score.shape[-2] > 1: # Prefill phase
            #if self.flag is not True:
             #   return past_key_values
            assert self.flag is True # only use for the first time

            for idx in range(attn_score.shape[-2]):
                cur_score = attn_score[:, idx, :idx + 1]
                self.score_sum[:, :cur_score.shape[-1]] += cur_score

                if text_mask[idx].item():
                    conn_score = cur_score * image_mask[:idx + 1].unsqueeze(0).float()
                else:
                    conn_score = cur_score * text_mask[:idx + 1].unsqueeze(0).float()


                self.conn_score_sum[:, :cur_score.shape[-1]] += conn_score
        else: # Decode phase
        # pass
                # 1. Update cumulative attention scores using H2O strategy
                #self.score_sum[:, :seq_len] += attn_score.squeeze(1)
                # Add decay factor (e.g., 0.95) to balance between old Heavy-Hitters and recent context
            self.score_sum[:, :seq_len] = (self.score_sum[:, :seq_len] * 0.95) + attn_score.squeeze(1)

            #for connection
            #if text_mask[idx].item():
             #       conn_score = cur_score * image_mask[:idx + 1].unsqueeze(0).float()
            #else:
             #   conn_score = cur_score * text_mask[:idx + 1].unsqueeze(0).float()

            #self.conn_score_sum[:, :cur_score.shape[-1]] += conn_score


        #forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        # Calculate how many tokens to keep based on the SAME ratio
        target_cache_size = int(num_of_token * (1 - self.ratio))
        forget_num = seq_len - target_cache_size

        if forget_num <= 0:
            return past_key_values

        if forget_num > 1:
            assert self.flag is True
            #if self.flag is not True:
            #    return past_key_values
            #self.flag = False # Note: We remove the 'self.flag' lock to allow continuous updates during decoding # self.flag = False

            # 2. Dynamic Eviction (Run every time forget_num > 0)
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

                #order = torch.argsort(combined_score)
                #throw_idx = order[:forget_num] + start
                #selected_idx = order[forget_num:] + start
                #Identify Heavy-Hitters based on updated score_sum
                sorted_indices = torch.argsort(combined_score)

                # Since we need to keep 'target_cache_size', we drop the lowest 'forget_num'
                # Adjust forget_num for the slicing range
                adj_forget_num = forget_num

                # Select tokens with high cumulative attention (Heavy-Hitters)
                selected_idx = torch.where(sorted_indices >= adj_forget_num)[0] + self.start_size
                throw_idx = torch.where(sorted_indices < adj_forget_num)[0] + self.start_size

                # === MODALITY-SEGREGATED MERGING (new) ===
                # Segregation only used on the first merge round -- see the
                # _first_merge_done comment in __init__ for why later rounds
                # always fall back to unconstrained merging instead.
                if self.segregate_merge and not self._first_merge_done:
                    merge_idx = self._nearest_same_modality_grid(throw_idx, selected_idx, text_mask, image_mask)
                else:
                    # original unconstrained behavior, unchanged ("3. Elastic Merging")
                    merge_idx = []
                    for i in range(len(throw_idx)):
                        nearest_idx = torch.abs(selected_idx - throw_idx[i]).argmin()
                        merge_idx.append(selected_idx[nearest_idx].unsqueeze(0))
                    merge_idx = torch.cat(merge_idx)
                # === END MODALITY-SEGREGATED MERGING ===

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


            # === SINGLE-TURN-SAFE MODALITY SEGREGATION (new) ===
            # Mark that a merge round has now happened -- any FURTHER
            # merge round in this session (i.e. a later __call__ that
            # reaches this branch again) will fall back to unconstrained
            # merging above, since image_start/image_token_len no longer
            # describe the current (already-compacted) cache layout.
            self._first_merge_done = True
            # === END SINGLE-TURN-SAFE MODALITY SEGREGATION ===
            if self.distance > 0:
                self.selected_idx = self.distance
            else:
                self.selected_idx = seq_len - forget_num + self.distance

            past_key_values_return = []

            for layer_idx, (k, v) in enumerate(past_key_values):
                #selected_idx = selected_idx_all[layer_idx]
                #merge_idx = merge_idx_all[layer_idx]
                #throw_idx = throw_idx_all[layer_idx]
                selected_idx = selected_idx_all[layer_idx].to(k.device)
                merge_idx = merge_idx_all[layer_idx].to(k.device)
                throw_idx = throw_idx_all[layer_idx].to(k.device)

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

        # 4. Critical: Update score_sum to match the new cache size/order
        # Since indices shifted, we should align score_sum for the next step
        new_score_sum = torch.zeros(self.layer_num, target_cache_size + 1).to(self.combined_sum.device)
        for idx in range(self.layer_num):
            new_score_sum[idx, :len(selected_idx_all[idx])] = self.score_sum[idx, selected_idx_all[idx]]
            self.combined_sum = new_score_sum

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
            for idx in range(attn_score.shape[-2]): #check again fro (-2)
                cur_score = attn_score[:, idx, :idx+1]
                self.score_sum[:, :(cur_score.shape[-1])] += cur_score
        else:
            pass

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        if forget_num <= 0:
            return past_key_values
        else:
            if forget_num > 1:
                assert self.flag is True
                # if self.flag is not True: #added by tabassum
                #     return past_key_values #added by tabassum
                #self.flag = False

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
            for idx in range(attn_score.shape[-2]): #updated by tabassum -1 to -2 to iterate over query len instead of key_len
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
                #self.flag = False
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


class HelasticCache:
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

        # # update attn score
        # 1. Update Attention Score (H2O with Decay)
        attn_score = torch.cat(attn_score, dim=0)
        attn_score = attn_score.mean(dim=1, keepdim=False)

        # [Fix] Expand score_sum if the current sequence is longer than the buffer
        if seq_len > self.score_sum.shape[-1]: 
            new_score_sum = torch.zeros(self.layer_num, seq_len + 1).to(self.score_sum.device)
            new_score_sum[:, :self.score_sum.shape[-1]] = self.score_sum
            self.score_sum = new_score_sum

        if attn_score.shape[-2] > 1: # Prefill phase
            assert self.flag is True # only use for the first time
            for idx in range(attn_score.shape[-2]): #check later about (-2)
                cur_score = attn_score[:, idx, :idx+1]
                self.score_sum[:, :(cur_score.shape[-1])] += cur_score
        else: # Decode phase
            # pass
                # 1. Update cumulative attention scores using H2O strategy
                #self.score_sum[:, :seq_len] += attn_score.squeeze(1)
                # Add decay factor (e.g., 0.95) to balance between old Heavy-Hitters and recent context
            self.score_sum[:, :seq_len] = (self.score_sum[:, :seq_len] * 0.95) + attn_score.squeeze(1)

        # Calculate how many tokens to keep based on the SAME ratio
        target_cache_size = int(num_of_token * (1 - self.ratio))
        forget_num = seq_len - target_cache_size

        if forget_num <= 0:
            return past_key_values

        if forget_num > 1:
            assert self.flag is True
# Note: We remove the 'self.flag' lock to allow continuous updates during decoding
# self.flag = False

        # 2. Dynamic Eviction (Run every time forget_num > 0)
        selected_idx_all = []
        merge_idx_all = []
        throw_idx_all = []
        for idx in range(self.layer_num):
# selected_idx = torch.where(torch.argsort(self.score_sum[idx, self.start_size:(seq_len - self.protect_size)]) > forget_num)[0] + self.start_size
# throw_idx = torch.where(torch.argsort(self.score_sum[idx, self.start_size:(seq_len - self.protect_size)]) <= forget_num)[0]

# Identify Heavy-Hitters (HH) based on cumulative scores
# Target range for scoring (excluding start_size and protect_size)
            target_scores = self.score_sum[idx, self.start_size:(seq_len - self.protect_size)]
# Identify Heavy-Hitters based on updated score_sum
            sorted_indices = torch.argsort(target_scores)

# Since we need to keep 'target_cache_size', we drop the lowest 'forget_num'
# Adjust forget_num for the slicing range
            adj_forget_num = forget_num

# Select tokens with high cumulative attention (Heavy-Hitters)
            selected_idx = torch.where(sorted_indices >= adj_forget_num)[0] + self.start_size
            throw_idx = torch.where(sorted_indices < adj_forget_num)[0] + self.start_size

# 3. Elastic Merging: Find nearest selected token for each thrown token
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
            # selected_idx = selected_idx_all[idx]
            #merge_idx = merge_idx_all[idx]
            # throw_idx = throw_idx_all[idx]
            selected_idx = selected_idx_all[idx].to(k.device)
            merge_idx = merge_idx_all[idx].to(k.device)
            throw_idx = throw_idx_all[idx].to(k.device)

            k_forget = k.gather(dim=-2, index=throw_idx.view(1,1,-1,1).expand(k.shape[0], k.shape[1], -1 ,k.shape[-1]))
            v_forget = v.gather(dim=-2, index=throw_idx.view(1,1,-1,1).expand(v.shape[0], v.shape[1], -1 ,v.shape[-1]))

            k = k.scatter_reduce(-2, merge_idx.view(1,1,-1,1).expand(k.shape[0], k.shape[1], -1 ,k.shape[-1]), k_forget, 'mean')
            v = v.scatter_reduce(-2, merge_idx.view(1,1,-1,1).expand(v.shape[0], v.shape[1], -1 ,v.shape[-1]), v_forget, 'mean')

            k_new = k.gather(dim=-2, index=selected_idx.view(1,1,-1,1).expand(k.shape[0], k.shape[1], -1 ,k.shape[-1]))
            v_new = v.gather(dim=-2, index=selected_idx.view(1,1,-1,1).expand(v.shape[0], v.shape[1], -1 ,v.shape[-1]))

            past_key_values_return.append([k_new, v_new])

# 4. Critical: Update score_sum to match the new cache size/order
# Since indices shifted, we should align score_sum for the next step
        new_score_sum = torch.zeros(self.layer_num, target_cache_size + 1).to(self.score_sum.device)
        for idx in range(self.layer_num):
            new_score_sum[idx, :len(selected_idx_all[idx])] = self.score_sum[idx, selected_idx_all[idx]]
        self.score_sum = new_score_sum

        return past_key_values_return