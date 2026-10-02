import torch


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
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

        self.score_sum = torch.zeros(layer_num, self.cache_size + 1, device=_device())
        self.conn_score_sum = torch.zeros(layer_num, self.cache_size + 1, device=_device())

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
                        torch.arange(self.start_size, device=self.score_sum.device),
                        selected_idx,
                        torch.tensor([seq_len - self.protect_size], device=self.score_sum.device),
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

        self.score_sum = torch.zeros(layer_num, self.cache_size + 1, device=_device())
        self.conn_score_sum = torch.zeros(layer_num, self.cache_size + 1, device=_device())

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
                        torch.arange(self.start_size, device=self.score_sum.device),
                        selected_idx,
                        torch.tensor([seq_len - self.protect_size], device=self.score_sum.device),
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

        self.score_sum = torch.zeros(layer_num, self.cache_size + 1, device=_device())
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

                    selected_idx = torch.cat([torch.arange(self.start_size, device=self.score_sum.device), selected_idx, torch.tensor([seq_len - self.protect_size], device=self.score_sum.device)], dim=0) # the last token is always kept

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

        self.score_sum = torch.zeros(self.cache_size + 1, device=_device())
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
                selected_idx = torch.cat([selected_idx, torch.arange(seq_len - self.protect_size, seq_len, device=self.score_sum.device)], dim=0)
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

        self.score_sum = torch.zeros(layer_num, self.cache_size + 1, device=_device())
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

            selected_idx = torch.cat([torch.arange(self.start_size, device=self.score_sum.device), selected_idx, torch.tensor([seq_len - self.protect_size], device=self.score_sum.device)], dim=0) # the last token is always kept

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
class _MultimodalCacheBase:
    """Shared, corrected bookkeeping for multimodal Elastic-style caches."""

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
        image_token_len=0,
        decay=None,
    ):
        # Number of tokens at the beginning that are never considered for
        # score-based eviction.  This follows the original implementation.
        self.start_size = start_size

        # Kept for compatibility with the original constructor / experiments.
        self.recent_size = recent_size

        # KV sequence dimensions used by LLaVA/LLaMA-style caches.
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim

        # Fraction of the logical uncompressed stream that should be removed.
        self.ratio = ratio

        # Protect the final KV entry from score-based eviction.
        self.protect_size = 1

        # Kept for compatibility with the original implementation.
        self.distance = distance

        # Number of transformer layers.  Each layer may select different KV
        # entries, so scores and modality masks MUST be layer-specific.
        self.layer_num = layer_num

        # Weight of ordinary incoming-attention importance.
        self.attn_weight = attn_weight

        # Weight of the visual-grounding bonus for TEXT tokens.
        self.conn_weight = conn_weight

        # Location of the ACTUAL expanded visual-token block in the first
        # model prefill.  The fixed evaluator infers this from the expanded KV.
        self.image_start = image_start
        self.image_token_len = image_token_len

        # decay=None  -> cumulative Elastic score
        # decay=0.95  -> recency-weighted H-Elastic score
        self.decay = decay

        # State tensors are allocated lazily on the attention tensor's device.
        # This avoids hard-coded .cuda() calls and keeps device handling sane.
        self.score_sum = None
        self.conn_score_sum = None

        # True at a cache position means that KV slot is an IMAGE slot.
        # Shape: [layer_num, current_physical_cache_length].
        self.image_mask = None

        # The first call receives the full multimodal prefill.  Every later
        # token in the intended workload is text-only.
        self.initialized = False

    # ------------------------------------------------------------------
    # State initialization / growth
    # ------------------------------------------------------------------
    def reset(self):
        """Reset all state at the beginning of a NEW dataset conversation."""
        self.score_sum = None
        self.conn_score_sum = None
        self.image_mask = None
        self.initialized = False

    def _initial_image_mask(self, seq_len, device):
        """Build the image mask for the FIRST, unpruned multimodal prefill."""
        mask = torch.zeros(
            self.layer_num, seq_len, dtype=torch.bool, device=device
        )

        if self.image_start is None or self.image_token_len <= 0:
            return mask

        start = max(0, int(self.image_start))
        end = min(seq_len, start + int(self.image_token_len))

        if start < end:
            mask[:, start:end] = True

        return mask

    def _initialize_state(self, seq_len, device):
        """Allocate scores and modality masks for the first model call."""
        self.score_sum = torch.zeros(
            self.layer_num, seq_len, device=device, dtype=torch.float32
        )
        self.conn_score_sum = torch.zeros_like(self.score_sum)
        self.image_mask = self._initial_image_mask(seq_len, device)
        self.initialized = True

    def _grow_state_for_new_text(self, seq_len):
        """
        Extend score/mask state when Q2/A2/Q3/... append new TEXT tokens.

        After pruning, the physical KV cache is shorter than the logical
        conversation.  When the model appends q new tokens, seq_len grows by q.
        These new entries are text, so their image-mask values are False.
        """
        current_len = self.score_sum.shape[1]

        if seq_len < current_len:
            raise RuntimeError(
                f"Cache state length ({current_len}) is larger than model KV "
                f"length ({seq_len}). Score/KV bookkeeping is inconsistent."
            )

        if seq_len == current_len:
            return

        add_len = seq_len - current_len
        device = self.score_sum.device

        self.score_sum = torch.cat(
            [
                self.score_sum,
                torch.zeros(
                    self.layer_num,
                    add_len,
                    device=device,
                    dtype=self.score_sum.dtype,
                ),
            ],
            dim=1,
        )

        self.conn_score_sum = torch.cat(
            [
                self.conn_score_sum,
                torch.zeros(
                    self.layer_num,
                    add_len,
                    device=device,
                    dtype=self.conn_score_sum.dtype,
                ),
            ],
            dim=1,
        )

        # All tokens appended after the first multimodal prefill are text.
        self.image_mask = torch.cat(
            [
                self.image_mask,
                torch.zeros(
                    self.layer_num,
                    add_len,
                    device=device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )

    # ------------------------------------------------------------------
    # Attention / multimodal score updates
    # ------------------------------------------------------------------
    def _prepare_attention(self, attentions):
        """
        Convert Hugging Face attention output to [layers, queries, keys].

        Each element is normally [batch=1, heads, query_len, key_len].
        Concatenating over the list dimension gives layers in dimension 0;
        averaging over heads leaves one attention matrix per layer.
        """
        if attentions is None:
            raise ValueError(
                "attentions is None. Run the model with output_attentions=True."
            )

        attn = torch.cat([a for a in attentions], dim=0)
        # Keep the model dtype here. Casting the complete [L,Q,K] tensor to
        # float32 is expensive on MPS and unnecessary; only the much smaller
        # reduced score tensors need float32 accumulation.
        return attn.mean(dim=1)  # [layers, query_len, key_len]

    def _update_scores(self, attn):
        """
        Update ordinary attention importance and text->image connection score.

        RAW SCORE (key-side importance)
        --------------------------------
        For each KV key, sum attention received from the current query tokens.
        This naturally rewards image keys when text queries attend to them.

        CONNECTION SCORE (query-side visual grounding)
        ------------------------------------------------
        For each TEXT query token q:

            conn(q) = sum_{k is image} Attention(q -> k)

        That scalar is written at q's OWN KV position.  Therefore a text token
        that strongly uses the image gets an extra retention bonus later.

        This is intentionally NOT "give image keys another connection bonus".
        Their text->image importance is already represented in raw_score.
        """
        num_layers, query_len, key_len = attn.shape

        if num_layers != self.layer_num:
            raise RuntimeError(
                f"Expected {self.layer_num} attention layers, got {num_layers}."
            )

        if key_len != self.score_sum.shape[1]:
            raise RuntimeError(
                f"Attention key length ({key_len}) != score length "
                f"({self.score_sum.shape[1]})."
            )

        # Sum the attention received by every key from the CURRENT query block.
        incoming = attn.sum(dim=1).float()  # [layers, key_len]

        if self.decay is None:
            # MMElastic: cumulative importance; old evidence is never decayed.
            self.score_sum += incoming
        else:
            # MMHelastic: recent attention gradually replaces stale history.
            self.score_sum.mul_(self.decay).add_(incoming)
            self.conn_score_sum.mul_(self.decay)

        # Query rows correspond to the LAST query_len positions in key space.
        # - first prefill: query_len == key_len -> positions 0..key_len-1
        # - later turn/decode: query positions are newly appended tail entries
        query_start = key_len - query_len

        # Vectorized equivalent of the former layer/query Python loops:
        # sum each query's attention to image keys, then write that grounding
        # score only at text-query positions in the KV tail. This avoids device
        # boolean checks and synchronizations for every layer and token.
        image_keys = self.image_mask.to(attn.dtype).unsqueeze(-1)
        image_attention = torch.bmm(attn, image_keys).squeeze(-1).float()
        query_is_text = ~self.image_mask[:, query_start:key_len]
        self.conn_score_sum[:, query_start:key_len].add_(
            image_attention * query_is_text
        )

    # ------------------------------------------------------------------
    # Score normalization / selection
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_nonnegative(values):
        """Normalize non-negative attention scores without NaNs."""
        if values.numel() == 0:
            return values
        max_value = values.max()
        if max_value <= 0:
            return torch.zeros_like(values)
        return values / (max_value + 1e-6)

    def _combined_scores(self, layer_idx, start, end):
        """Return eviction scores for the candidate interval [start, end)."""
        raw = self.score_sum[layer_idx, start:end]
        conn = self.conn_score_sum[layer_idx, start:end]
        image = self.image_mask[layer_idx, start:end]
        text = ~image

        raw_norm = self._normalize_nonnegative(raw)
        conn_norm = torch.zeros_like(conn)

        # Connection score is meaningful only for text tokens.
        if text.any():
            conn_norm[text] = self._normalize_nonnegative(conn[text])

        combined = self.attn_weight * raw_norm
        combined[text] += self.conn_weight * conn_norm[text]
        return combined

    def _ensure_image_survives(
        self,
        layer_idx,
        start,
        candidate_scores,
        selected_rel,
        throw_rel,
    ):
        """
        Ensure same-modality image merging remains possible.

        If image candidates exist and an eviction would remove ALL of them,
        retain the highest-scoring image candidate and instead evict the
        lowest-scoring selected text candidate.  The number of retained entries
        stays unchanged.
        """
        candidate_image = self.image_mask[layer_idx, start:start + len(candidate_scores)]

        # Nothing special is needed if there are no image candidates.
        if not candidate_image.any():
            return selected_rel, throw_rel

        # If at least one image token already survives, we are done.
        if selected_rel.numel() > 0 and candidate_image[selected_rel].any():
            return selected_rel, throw_rel

        # Choose the best image token from the throw set to rescue.
        thrown_images = throw_rel[candidate_image[throw_rel]]
        if thrown_images.numel() == 0:
            return selected_rel, throw_rel

        rescue_rel = thrown_images[
            torch.argmax(candidate_scores[thrown_images])
        ]

        # To keep cache size unchanged, swap out the lowest-scoring selected
        # TEXT candidate.  Start/protected tokens are outside this candidate
        # set and therefore remain untouched.
        selected_text = selected_rel[~candidate_image[selected_rel]]
        if selected_text.numel() == 0:
            # Extremely small/degenerate cache: cannot perform a size-neutral
            # swap.  Keep selection unchanged rather than corrupt modalities.
            return selected_rel, throw_rel

        evict_rel = selected_text[
            torch.argmin(candidate_scores[selected_text])
        ]

        selected_rel = selected_rel[selected_rel != evict_rel]
        selected_rel = torch.cat([selected_rel, rescue_rel.view(1)])

        throw_rel = throw_rel[throw_rel != rescue_rel]
        throw_rel = torch.cat([throw_rel, evict_rel.view(1)])

        return selected_rel, throw_rel

    def _same_modality_merge_targets(
        self, layer_idx, throw_idx, retained_idx
    ):
        """
        Map every forgotten token to the nearest retained token of SAME modality.

        Because image/text masks are hard labels, cross-modality merging would
        create a mixed KV vector that cannot truthfully be called either text or
        image.  We therefore forbid such merges.
        """
        if throw_idx.numel() == 0:
            return throw_idx.clone()

        layer_image = self.image_mask[layer_idx]
        targets = []

        for forgotten in throw_idx:
            forgotten_is_image = layer_image[forgotten]
            same_modality = retained_idx[
                layer_image[retained_idx] == forgotten_is_image
            ]

            if same_modality.numel() == 0:
                raise RuntimeError(
                    "No retained KV slot of the same modality is available for "
                    "merging. Reduce the pruning ratio or retain at least one "
                    "image/text token."
                )

            nearest = torch.argmin(torch.abs(same_modality - forgotten))
            targets.append(same_modality[nearest])

        return torch.stack(targets)

    # ------------------------------------------------------------------
    # Pruning / merging / state realignment
    # ------------------------------------------------------------------
    def _merge_and_gather_layer(
        self,
        k,
        v,
        layer_idx,
        retained_idx,
        throw_idx,
        merge_idx,
    ):
        """Merge forgotten KV entries, then gather retained entries."""
        device = k.device
        retained_idx = retained_idx.to(device)
        throw_idx = throw_idx.to(device)
        merge_idx = merge_idx.to(device)

        # If nothing is being dropped, only gather (normally identity).
        if throw_idx.numel() == 0:
            k_new = k.index_select(self.k_seq_dim, retained_idx)
            v_new = v.index_select(self.v_seq_dim, retained_idx)
            return k_new, v_new

        # Gather forgotten K/V vectors.
        k_forget = k.index_select(self.k_seq_dim, throw_idx)
        v_forget = v.index_select(self.v_seq_dim, throw_idx)

        # scatter_reduce expects an index tensor with the same rank as K/V.
        k_index_shape = [1] * k.dim()
        k_index_shape[self.k_seq_dim] = -1
        k_index_shape[-1] = 1
        k_index = merge_idx.view(*k_index_shape).expand_as(k_forget)

        v_index_shape = [1] * v.dim()
        v_index_shape[self.v_seq_dim] = -1
        v_index_shape[-1] = 1
        v_index = merge_idx.view(*v_index_shape).expand_as(v_forget)

        # include_self=True (PyTorch default) averages the retained vector with
        # all forgotten vectors mapped to that slot.
        k_merged = k.scatter_reduce(
            self.k_seq_dim, k_index, k_forget, reduce="mean"
        )
        v_merged = v.scatter_reduce(
            self.v_seq_dim, v_index, v_forget, reduce="mean"
        )

        # Finally physically remove forgotten slots.
        k_new = k_merged.index_select(self.k_seq_dim, retained_idx)
        v_new = v_merged.index_select(self.v_seq_dim, retained_idx)
        return k_new, v_new

    def _realign_scores_and_masks(
        self,
        retained_idx_all,
        throw_idx_all,
        merge_idx_all,
    ):
        """
        Realign score_sum, conn_score_sum, and image_mask after KV pruning.

        Before gathering, score mass from a forgotten token is transferred to
        the same retained slot that receives its merged KV representation.
        Then all state tensors are gathered with each layer's retained indices.
        """
        new_scores = []
        new_conn_scores = []
        new_masks = []

        for layer_idx in range(self.layer_num):
            retained = retained_idx_all[layer_idx]
            thrown = throw_idx_all[layer_idx]
            merge = merge_idx_all[layer_idx]

            layer_score = self.score_sum[layer_idx].clone()
            layer_conn = self.conn_score_sum[layer_idx].clone()

            # Transfer the importance carried by forgotten entries into the
            # retained representation that absorbs them.
            if thrown.numel() > 0:
                layer_score.scatter_add_(0, merge, layer_score[thrown])
                layer_conn.scatter_add_(0, merge, layer_conn[thrown])

            new_scores.append(layer_score[retained])
            new_conn_scores.append(layer_conn[retained])
            new_masks.append(self.image_mask[layer_idx, retained])

        self.score_sum = torch.stack(new_scores, dim=0)
        self.conn_score_sum = torch.stack(new_conn_scores, dim=0)
        self.image_mask = torch.stack(new_masks, dim=0)

    def _prune(self, past_key_values, forget_num):
        """Score, same-modality merge, gather, and realign all cache state."""
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        # Candidate interval excludes protected beginning and final token.
        start = min(self.start_size, seq_len)
        end = max(start, seq_len - self.protect_size)
        candidate_count = max(0, end - start)

        if candidate_count == 0 or forget_num <= 0:
            return past_key_values

        # We cannot drop more than the score-based candidate interval.
        drop_count = min(int(forget_num), candidate_count)

        retained_idx_all = []
        throw_idx_all = []
        merge_idx_all = []

        device = self.score_sum.device

        for layer_idx in range(self.layer_num):
            combined = self._combined_scores(layer_idx, start, end)
            order = torch.argsort(combined)  # low score -> high score

            throw_rel = order[:drop_count]
            selected_rel = order[drop_count:]

            # If pruning would eliminate every image candidate, swap one back
            # in so image->image merging remains valid.
            selected_rel, throw_rel = self._ensure_image_survives(
                layer_idx,
                start,
                combined,
                selected_rel,
                throw_rel,
            )

            selected_candidate = selected_rel + start
            thrown = throw_rel + start

            # Protected entries are retained regardless of their scores.
            prefix = torch.arange(0, start, device=device, dtype=torch.long)
            suffix = torch.arange(end, seq_len, device=device, dtype=torch.long)

            retained = torch.cat(
                [prefix, selected_candidate, suffix], dim=0
            )

            # Restore physical sequence order.  The score ranking must NOT
            # reorder the cache, because positional order still matters.
            retained = torch.sort(retained).values

            merge = self._same_modality_merge_targets(
                layer_idx, thrown, retained
            )

            retained_idx_all.append(retained)
            throw_idx_all.append(thrown)
            merge_idx_all.append(merge)

        # Apply the per-layer selections to K/V.
        new_past = []
        for layer_idx, (k, v) in enumerate(past_key_values):
            k_new, v_new = self._merge_and_gather_layer(
                k,
                v,
                layer_idx,
                retained_idx_all[layer_idx],
                throw_idx_all[layer_idx],
                merge_idx_all[layer_idx],
            )
            new_past.append([k_new, v_new])

        # Crucial: state arrays must now describe the NEW physical KV layout.
        self._realign_scores_and_masks(
            retained_idx_all, throw_idx_all, merge_idx_all
        )

        return new_past

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        """Update importance from this model call and prune to target size."""
        if past_key_values is None:
            return None

        if num_of_token is None:
            raise ValueError("num_of_token is required for ratio-based pruning.")

        attn = self._prepare_attention(attentions)
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        # Initialize the original image/text layout once, then append text-only
        # modality entries on all later calls.
        if not self.initialized:
            self._initialize_state(seq_len, attn.device)
        else:
            self._grow_state_for_new_text(seq_len)

        # Scores and masks now have exactly the same physical key dimension as
        # the model attention and KV cache.
        self._update_scores(attn)

        # num_of_token is the logical UNCOMPRESSED stream length maintained by
        # the evaluator.  The target grows as the conversation grows.
        target_cache_size = max(1, int(num_of_token * (1.0 - self.ratio)))
        forget_num = seq_len - target_cache_size

        if forget_num <= 0:
            return past_key_values

        return self._prune(past_key_values, forget_num)


class MultimodalElasticCache(_MultimodalCacheBase):
    """
    Corrected multi-turn MMElastic.

    Difference from MMHelastic:
      - MMElastic accumulates attention evidence over the entire conversation.
      - Old importance is NOT decayed.

    This is more stable/history-preserving, but less responsive to a shift in
    what later turns are discussing.
    """

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
        image_token_len=0,
    ):
        super().__init__(
            start_size=start_size,
            recent_size=recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=ratio,
            distance=distance,
            layer_num=layer_num,
            attn_weight=attn_weight,
            conn_weight=conn_weight,
            image_start=image_start,
            image_token_len=image_token_len,
            decay=None,
        )


class MultimodalHelasticCache(_MultimodalCacheBase):
    """
    Corrected multi-turn MMHelastic.

    Difference from MMElastic:
      - MMHelastic multiplies old importance by `decay` before adding the
        current attention evidence.
      - With decay=0.95, recent turns gradually outweigh stale early evidence.

    This is the dynamic/adaptive version for evolving multi-turn context.
    """

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
        image_token_len=0,
        decay=0.95,
    ):
        super().__init__(
            start_size=start_size,
            recent_size=recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=ratio,
            distance=distance,
            layer_num=layer_num,
            attn_weight=attn_weight,
            conn_weight=conn_weight,
            image_start=image_start,
            image_token_len=image_token_len,
            decay=decay,
        )


class TurnIsolatedMultimodalCache(_MultimodalCacheBase):
    """MMElastic variant that compresses once per turn and freezes survivors.

    Calls to ``__call__`` only collect attention evidence. ``finalize_turn``
    independently compresses the active image/text segments (image exists only
    in turn 1), while entries retained from completed turns never compete or
    merge with entries from a later turn.
    """

    def __init__(self, *args, initial_protected_text_len=0, **kwargs):
        kwargs["decay"] = None
        super().__init__(*args, **kwargs)
        self.initial_protected_text_len = max(0, int(initial_protected_text_len))
        self.frozen_mask = None
        self.turn_index = 0
        self.last_finalize_stats = None

    def reset(self):
        super().reset()
        self.frozen_mask = None
        self.turn_index = 0
        self.last_finalize_stats = None

    def _initialize_state(self, seq_len, device):
        super()._initialize_state(seq_len, device)
        self.frozen_mask = torch.zeros(
            self.layer_num, seq_len, dtype=torch.bool, device=device
        )

    def _grow_state_for_new_text(self, seq_len):
        old_len = self.score_sum.shape[1]
        super()._grow_state_for_new_text(seq_len)
        add_len = seq_len - old_len
        if add_len > 0:
            self.frozen_mask = torch.cat(
                [
                    self.frozen_mask,
                    torch.zeros(
                        self.layer_num,
                        add_len,
                        dtype=torch.bool,
                        device=self.frozen_mask.device,
                    ),
                ],
                dim=1,
            )

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        attn = self._prepare_attention(attentions)
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if not self.initialized:
            self._initialize_state(seq_len, attn.device)
        else:
            self._grow_state_for_new_text(seq_len)
        self._update_scores(attn)
        return past_key_values

    def _keep_count(self, count):
        if count == 0:
            return 0
        return max(1, int(count * (1.0 - self.ratio)))

    @staticmethod
    def _top_keep(indices, scores, keep_count):
        if keep_count == 0:
            return indices[:0]
        relative = torch.topk(scores, k=min(keep_count, indices.numel())).indices
        return torch.sort(indices[relative]).values

    @staticmethod
    def _nearest_targets(thrown, kept):
        if thrown.numel() == 0:
            return thrown.clone()
        if kept.numel() == 0:
            raise RuntimeError("Turn isolation cannot merge into an empty segment.")
        distances = torch.abs(thrown[:, None] - kept[None, :])
        return kept[torch.argmin(distances, dim=1)]

    def _select_image_keep(self, image_candidates, image_scores, keep_count):
        """Select active visual entries; base v2 uses attention importance."""
        return self._top_keep(image_candidates, image_scores, keep_count)

    def _visual_policy_name(self):
        return "turn1_attention_ranked" if self.turn_index == 0 else "already_frozen"

    def finalize_turn(self, past_key_values):
        """Compress the active turn once, then mark every survivor frozen."""
        if past_key_values is None or not self.initialized:
            return past_key_values
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if self.score_sum.shape[1] != seq_len:
            raise RuntimeError(
                f"Turn-finalization state length {self.score_sum.shape[1]} "
                f"does not match KV length {seq_len}."
            )

        retained_all, thrown_all, merge_all = [], [], []
        stats = None
        for layer_idx in range(self.layer_num):
            device = self.score_sum.device
            indices = torch.arange(seq_len, device=device)
            frozen = self.frozen_mask[layer_idx]
            image = self.image_mask[layer_idx]
            protected = torch.zeros(seq_len, dtype=torch.bool, device=device)
            if self.turn_index == 0:
                prefix_keep = max(
                    self.start_size, self.initial_protected_text_len
                )
            else:
                prefix_keep = self.start_size
            protected[: min(prefix_keep, seq_len)] = True
            active = ~frozen
            candidates = active & ~protected
            image_candidates = indices[candidates & image]
            text_candidates = indices[candidates & ~image]

            image_scores = self._normalize_nonnegative(
                self.score_sum[layer_idx, image_candidates]
            )
            text_scores = (
                self.attn_weight
                * self._normalize_nonnegative(
                    self.score_sum[layer_idx, text_candidates]
                )
                + self.conn_weight
                * self._normalize_nonnegative(
                    self.conn_score_sum[layer_idx, text_candidates]
                )
            )
            image_keep = self._select_image_keep(
                image_candidates, image_scores, self._keep_count(image_candidates.numel())
            )
            text_keep = self._top_keep(
                text_candidates, text_scores, self._keep_count(text_candidates.numel())
            )
            fixed_keep = indices[frozen | (active & protected)]
            retained = torch.unique(
                torch.cat([fixed_keep, image_keep, text_keep]), sorted=True
            )
            image_throw = image_candidates[~torch.isin(image_candidates, image_keep)]
            text_throw = text_candidates[~torch.isin(text_candidates, text_keep)]
            thrown = torch.cat([image_throw, text_throw])
            merge = torch.cat(
                [
                    self._nearest_targets(image_throw, image_keep),
                    self._nearest_targets(text_throw, text_keep),
                ]
            )
            if thrown.numel():
                order = torch.argsort(thrown)
                thrown, merge = thrown[order], merge[order]
            retained_all.append(retained)
            thrown_all.append(thrown)
            merge_all.append(merge)
            if stats is None:
                stats = {
                    "turn": self.turn_index + 1,
                    "kv_before": seq_len,
                    "frozen_before": int(frozen.sum().item()),
                    "protected_prefix": int(min(prefix_keep, seq_len)),
                    "visual_policy": self._visual_policy_name(),
                    "image_candidates": int(image_candidates.numel()),
                    "image_kept": int(image_keep.numel()),
                    "text_candidates": int(text_candidates.numel()),
                    "text_kept": int(text_keep.numel()),
                    "kv_after": int(retained.numel()),
                }

        new_past = []
        for layer_idx, (key, value) in enumerate(past_key_values):
            key, value = self._merge_and_gather_layer(
                key,
                value,
                layer_idx,
                retained_all[layer_idx],
                thrown_all[layer_idx],
                merge_all[layer_idx],
            )
            new_past.append([key, value])
        self._realign_scores_and_masks(retained_all, thrown_all, merge_all)
        self.frozen_mask = torch.ones_like(self.image_mask)
        self.turn_index += 1
        self.last_finalize_stats = stats
        return new_past


class PersistentVisualTurnIsolatedCache(TurnIsolatedMultimodalCache):
    """Turn-isolated MMElastic with question-independent visual anchors.

    Turn 1 keeps uniformly spaced entries over the expanded visual-token block,
    merges discarded visual entries into their nearest retained visual anchor,
    and freezes those anchors for all later turns. Text selection remains the
    same attention-plus-grounding rule used by turn-isolated MMElastic.
    """

    @staticmethod
    def _coverage_keep(indices, keep_count):
        count = int(indices.numel())
        keep_count = int(keep_count)
        if count == 0 or keep_count <= 0:
            return indices[:0]
        if keep_count >= count:
            return indices.clone()
        if keep_count == 1:
            return indices[count // 2 : count // 2 + 1]

        positions = torch.linspace(
            0,
            count - 1,
            steps=keep_count,
            device=indices.device,
            dtype=torch.float32,
        ).round().long()
        positions = torch.unique(positions, sorted=True)
        if positions.numel() < keep_count:
            all_positions = torch.arange(count, device=indices.device)
            unused = all_positions[~torch.isin(all_positions, positions)]
            positions = torch.sort(
                torch.cat([positions, unused[: keep_count - positions.numel()]])
            ).values
        return indices[positions]

    def _select_image_keep(self, image_candidates, image_scores, keep_count):
        del image_scores
        return self._coverage_keep(image_candidates, keep_count)

    def _visual_policy_name(self):
        return (
            "uniform_persistent_coverage"
            if self.turn_index == 0
            else "already_frozen"
        )
"""TGV-KV adapted to the legacy Transformers 4.31 LLaVA implementation.

The cache policy follows the official TGV-KV implementation vendored under
``third_party/TGV-KV``.  Two compatibility changes are necessary here:

* legacy LLaVA exposes one negative image placeholder which expands internally,
  so the evaluator supplies the actual expanded visual-token span;
* TGV-KV's token-preservation reallocation gives layers different cache
  lengths.  Transformers 4.31 assumes all layers have the first layer's cache
  length, so ``enable_legacy_tgv_attention`` makes its LLaMA attention accept
  the correct mask length for each layer.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)


_ORIGINAL_LLAMA_ATTENTION_FORWARD = LlamaAttention.forward


def _tgv_compatible_llama_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
):
    """Transformers 4.31 LLaMA attention with unequal-layer-cache support."""
    bsz, q_len, _ = hidden_states.size()

    if self.pretraining_tp > 1:
        kv_slice = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
        q_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.pretraining_tp, dim=0
        )
        k_slices = self.k_proj.weight.split(kv_slice, dim=0)
        v_slices = self.v_proj.weight.split(kv_slice, dim=0)
        query_states = torch.cat(
            [F.linear(hidden_states, weight) for weight in q_slices], dim=-1
        )
        key_states = torch.cat(
            [F.linear(hidden_states, weight) for weight in k_slices], dim=-1
        )
        value_states = torch.cat(
            [F.linear(hidden_states, weight) for weight in v_slices], dim=-1
        )
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # position_ids are derived from the first layer by legacy LlamaModel.  A
    # more heavily pruned later layer can be shorter, so its rotary table must
    # still cover that shared logical position.
    # Legacy LlamaModel constructs this mask from layer 0. Its shape is already
    # a host-side integer and covers the shared position_ids. Using
    # position_ids.max().item() here would force an MPS/CUDA synchronization in
    # every layer for every generated token, making multi-turn decoding appear
    # to hang.
    rotary_seq_len = max(
        kv_seq_len, int(getattr(self, "_logical_position_limit", 0))
    )
    if attention_mask is not None:
        rotary_seq_len = max(rotary_seq_len, attention_mask.shape[-1])
    cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    new_past = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        # The mask is constructed using layer 0's cache length.  Past tokens
        # are unmasked in this batch-size-one evaluator, so retain the causal
        # tail and resize the all-zero past portion for this layer.
        if attention_mask.shape[-1] >= kv_seq_len:
            layer_mask = attention_mask[..., -kv_seq_len:]
        else:
            padding = torch.zeros(
                (*attention_mask.shape[:-1], kv_seq_len - attention_mask.shape[-1]),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            layer_mask = torch.cat([padding, attention_mask], dim=-1)
        attn_weights = attn_weights + layer_mask

    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(
        bsz, q_len, self.hidden_size
    )

    if self.pretraining_tp > 1:
        chunks = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
        weights = self.o_proj.weight.split(
            self.hidden_size // self.pretraining_tp, dim=1
        )
        attn_output = sum(F.linear(chunks[i], weights[i]) for i in range(self.pretraining_tp))
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, new_past


def enable_legacy_tgv_attention():
    """Enable the compatibility patch once, only for a TGV-KV evaluation."""
    if LlamaAttention.forward is not _tgv_compatible_llama_attention_forward:
        LlamaAttention.forward = _tgv_compatible_llama_attention_forward


def set_legacy_logical_position_limit(model, logical_length):
    """Size every legacy LLaMA RoPE table for an explicit logical position."""
    inner = model.get_model() if hasattr(model, "get_model") else model.model
    for layer in inner.layers:
        layer.self_attn._logical_position_limit = int(logical_length)


class TGVKVCache:
    """Official TGV-KV scoring/allocation adapted to expanded LLaVA images."""

    def __init__(
        self,
        layer_num,
        image_start,
        image_token_len,
        start_size=4,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.0,
        batch_size=1,
    ):
        self.layer_num = layer_num
        self.image_start = int(image_start)
        self.image_token_len = int(image_token_len)
        self.start_size = start_size
        self.protect_size = 1
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.ratio = ratio
        self.batch_size = batch_size
        self.ratios = np.full(layer_num, ratio, dtype=np.float32)
        self.initial_text_len_list = [self.protect_size] * layer_num

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        if attentions is None:
            raise ValueError("TGV-KV requires output_attentions=True.")
        if attentions[0].shape[-2] > 1:
            return self._prefill(past_key_values, num_of_token, attentions)
        return self._decode(past_key_values, num_of_token, attentions)

    def _prefill(self, past_key_values, num_of_token, attentions):
        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        seq_len = int(seq_lens[0])
        total_forget = int(seq_len - num_of_token * (1 - self.ratio)) * self.layer_num
        if total_forget <= 0:
            return past_key_values

        image_start = self.image_start
        text_start = image_start + self.image_token_len
        all_attns = torch.stack([x.squeeze(0).mean(0).float() for x in attentions])
        text_image_attns = all_attns[:, text_start:, image_start:text_start]
        text_text_attns = all_attns[:, text_start:, text_start:]

        text_text_score = text_text_attns.sum(1, keepdim=True)
        divisor = torch.arange(
            1, text_text_score.shape[-1] + 1, device=text_text_score.device
        ).flip([0]).view(1, 1, -1)
        text_text_score = text_text_score / divisor
        text_image_score = (
            text_image_attns * text_text_score.transpose(-1, -2)
        ).sum(1, keepdim=True)
        pre_score = all_attns[..., :image_start].sum(1, keepdim=True) + 100
        post_score = all_attns[..., text_start:].sum(1, keepdim=True) + 100
        score_sum = torch.cat([pre_score, text_image_score, post_score], dim=2)

        layer_mass = text_image_attns.reshape(self.layer_num, -1).sum(dim=1)
        if float(layer_mass.sum().item()) <= 0:
            normalized = torch.full_like(layer_mass, 1.0 / self.layer_num)
        else:
            normalized = layer_mass / layer_mass.sum()
        layer_ratio = (
            seq_len - self.layer_num * seq_len * (1 - self.ratio) * normalized
        ) / seq_len
        self.ratios = layer_ratio.cpu().numpy()
        forget_nums = np.maximum((self.ratios * seq_lens).round().astype(np.int32), 0)

        result = []
        self.initial_text_len_list = []
        for idx, (k, v) in enumerate(past_key_values):
            forget_num = min(int(forget_nums[idx]), max(0, seq_len - self.start_size - self.protect_size))
            candidate = score_sum[idx, :, self.start_size : seq_len - self.protect_size]
            selected = torch.argsort(candidate, dim=1)[:, forget_num:] + self.start_size
            selected = selected.sort().values
            pre = torch.arange(self.start_size, device=selected.device).view(1, -1)
            post = torch.tensor([seq_len - 1], device=selected.device).view(1, -1)
            selected = torch.cat([pre, selected, post], dim=-1)
            self.initial_text_len_list.append(
                max(int((selected[0] >= text_start).sum().item()), self.protect_size)
            )
            gather = selected.to(k.device).view(self.batch_size, 1, -1, 1)
            result.append((
                k.gather(-2, gather.expand(-1, k.shape[1], -1, k.shape[-1])),
                v.gather(-2, gather.expand(-1, v.shape[1], -1, v.shape[-1])),
            ))
        return tuple(result)

    def _decode(self, past_key_values, num_of_token, attentions):
        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        forget_nums = (seq_lens - num_of_token * (1 - self.ratios)).astype(np.int32)
        forget_nums = np.maximum(forget_nums, 0)
        result = []
        for idx, (k, v) in enumerate(past_key_values):
            if forget_nums[idx] <= 0:
                result.append((k, v))
                continue
            seq_len = int(seq_lens[idx])
            evict_start = self.start_size
            evict_end = seq_len - self.initial_text_len_list[idx]
            if evict_start >= evict_end:
                result.append((k, v))
                continue
            score = attentions[idx].mean(1).squeeze(0).sum(0)
            # Keep the eviction index on the accelerator. Calling .item() here
            # creates one device-to-host synchronization per layer per token.
            pruned = score[evict_start:evict_end].argmin() + evict_start
            indices = torch.arange(seq_len, device=k.device)
            retained = indices[indices != pruned.to(k.device)]
            result.append((
                k.index_select(self.k_seq_dim, retained),
                v.index_select(self.v_seq_dim, retained),
            ))
        return tuple(result)
