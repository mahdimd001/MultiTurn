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

        self.ratio = ratio
        self.protect_size = 1
        self.distance = distance
        self.layer_num = layer_num

        self.reset()

    def reset(self):
        self.score_sum = torch.zeros(self.layer_num, self.cache_size + 1).cuda()
        self.flag = True
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
            # multi-turn: no one-shot gate, and score against the true cache
            # position (chunk_start+idx), not the position within this chunk
            chunk_start = seq_len - attn_score.shape[-2]
            for idx in range(attn_score.shape[-2]):
                pos = chunk_start + idx
                cur_score = attn_score[:, idx, :pos+1]
                self.score_sum[:, :(cur_score.shape[-1])] += cur_score
        else:
            pass

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        if forget_num <= 0:
            return past_key_values
        else:
            if forget_num > 1:

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
            

class MMHElasticCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.,
        distance=-25,
        layer_num=40,
        attn_weight=1.0,
        conn_weight=0.8,
        image_start=None,
        image_token_len=576,
        same_modality_merge=True,   # thrown tokens merge only into same-modality anchors
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

        self.ratio = ratio
        self.protect_size = 1
        self.distance = distance
        self.layer_num = layer_num

        self.attn_weight = attn_weight
        self.conn_weight = conn_weight

        self.image_start = image_start
        self.image_token_len = image_token_len
        self.same_modality_merge = same_modality_merge

        self.reset()

    def reset(self):
        self.score_sum = torch.zeros(self.layer_num, self.cache_size + 1).cuda()
        self.conn_score_sum = torch.zeros(self.layer_num, self.cache_size + 1).cuda()
        self.flag = True
        self.selected_idx = 0
        self.modality_is_image = None

    def _modality_masks(self, seq_len, device):
        text_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        image_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if self.image_start is None:
            return text_mask, image_mask
        image_end = min(self.image_start + self.image_token_len, seq_len)
        if self.image_start < seq_len:
            text_mask[self.image_start:image_end] = False
            image_mask[self.image_start:image_end] = True
        return text_mask, image_mask

    def _init_modality_state(self, seq_len, device):
        # first call only: built from the true, unmerged prompt layout
        is_image = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if self.image_start is not None and self.image_start < seq_len:
            image_end = min(self.image_start + self.image_token_len, seq_len)
            is_image[self.image_start:image_end] = True
        self.modality_is_image = is_image

    def _grow_modality_state(self, seq_len, device):
        # a later turn appended tokens; those are always text
        cur = self.modality_is_image.shape[0]
        if seq_len <= cur:
            return
        self.modality_is_image = torch.cat([
            self.modality_is_image,
            torch.zeros(seq_len - cur, dtype=torch.bool, device=device),
        ])

    def _compact_modality_state(self, selected_idx):
        # gather the modality vector the same way k/v are gathered
        self.modality_is_image = self.modality_is_image[selected_idx]

    def _grow(self, seq_len):
        if seq_len < self.score_sum.shape[-1]:
            return
        for name in ('score_sum', 'conn_score_sum'):
            old = getattr(self, name)
            new = torch.zeros(self.layer_num, seq_len + 1, device=old.device)
            new[:, :old.shape[-1]] = old
            setattr(self, name, new)

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        attn_score = [attention for attention in attentions]
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        self._grow(seq_len)

        # update attn score and cross-modal connection score
        attn_score = torch.cat(attn_score, dim=0)
        attn_score = attn_score.mean(dim=1, keepdim=False)
        if self.modality_is_image is None:
            self._init_modality_state(seq_len, attn_score.device)
        else:
            self._grow_modality_state(seq_len, attn_score.device)
        image_mask = self.modality_is_image
        text_mask = ~self.modality_is_image
        if attn_score.shape[-2] > 1:
            chunk_start = seq_len - attn_score.shape[-2]
            for idx in range(attn_score.shape[-2]):
                pos = chunk_start + idx
                cur_score = attn_score[:, idx, :pos+1]
                self.score_sum[:, :(cur_score.shape[-1])] += cur_score

                # a query attends across modalities: credit the keys it lands on
                if text_mask[pos]:
                    other = image_mask[:pos+1]
                else:
                    other = text_mask[:pos+1]
                self.conn_score_sum[:, :(cur_score.shape[-1])] += cur_score * other.unsqueeze(0).float()
        else:
            pass

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        if forget_num <= 0:
            return past_key_values
        else:
            if forget_num > 1:
                selected_idx_all = []
                merge_idx_all = []
                throw_idx_all = []
                for idx in range(self.layer_num):
                    raw_score = self.score_sum[idx, self.start_size:(seq_len - self.protect_size)]
                    conn_score = self.conn_score_sum[idx, self.start_size:(seq_len - self.protect_size)]
                    combined_score = self.attn_weight * (raw_score / (raw_score.max() + 1e-6)) \
                                   + self.conn_weight * (conn_score / (conn_score.max() + 1e-6))

                    selected_idx = torch.where(torch.argsort(combined_score) > forget_num)[0] + self.start_size
                    throw_idx = torch.where(torch.argsort(combined_score) <= forget_num)[0]
                    if self.same_modality_merge:
                        sel_is_image = self.modality_is_image[selected_idx]
                        img_anchors = selected_idx[sel_is_image]
                        txt_anchors = selected_idx[~sel_is_image]

                    merge_idx = []
                    for i in range(len(throw_idx)):
                        if self.same_modality_merge:
                            same = img_anchors if self.modality_is_image[throw_idx[i]] else txt_anchors
                            if same.numel() == 0:
                                same = selected_idx
                        else:
                            same = selected_idx
                        merge_idx.append(same[torch.abs((same - throw_idx[i])).argmin()].unsqueeze(0))
                    merge_idx = torch.cat(merge_idx)

                    selected_idx = torch.cat([torch.arange(self.start_size).cuda(), selected_idx, torch.tensor([seq_len - self.protect_size]).cuda()], dim=0) # the last token is always kept

                    selected_idx_all.append(selected_idx)
                    merge_idx_all.append(merge_idx)
                    throw_idx_all.append(throw_idx)

                self._compact_modality_state(selected_idx_all[0])

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