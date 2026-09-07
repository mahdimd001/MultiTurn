import torch
import torch.nn.functional as F


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _expand_scores(score_tensor, length):
    if score_tensor.shape[-1] >= length:
        return score_tensor
    new_shape = list(score_tensor.shape)
    new_shape[-1] = length
    new_score = torch.zeros(*new_shape, device=score_tensor.device, dtype=score_tensor.dtype)
    new_score[..., : score_tensor.shape[-1]] = score_tensor
    return new_score


def _gather_seq(x, idx, seq_dim):
    idx = idx.to(x.device).long()
    view_shape = [1] * x.dim()
    view_shape[seq_dim] = -1
    expand_shape = list(x.shape)
    expand_shape[seq_dim] = idx.numel()
    return x.gather(dim=seq_dim, index=idx.view(*view_shape).expand(*expand_shape))


def _scatter_reduce_seq(x, idx, src, seq_dim, reduce="mean"):
    idx = idx.to(x.device).long()
    view_shape = [1] * x.dim()
    view_shape[seq_dim] = -1
    expand_shape = list(src.shape)
    return x.scatter_reduce(
        dim=seq_dim,
        index=idx.view(*view_shape).expand(*expand_shape),
        src=src,
        reduce=reduce,
    )


def _cat_remove(x, remove_idx, seq_dim):
    seq_len = x.size(seq_dim)
    return torch.cat(
        [
            DIM_TO_SLICE[seq_dim](x, 0, remove_idx),
            DIM_TO_SLICE[seq_dim](x, remove_idx + 1, seq_len),
        ],
        dim=seq_dim,
    )


def _as_legacy_past(past_key_values):
    return tuple(x for x in past_key_values)


def _mean_attention_per_layer(attentions):
    # input: tuple/list of layer attentions, each [batch, heads, query_len, key_len]
    # batch is expected to be 1 in this script.
    attn_score = torch.cat([a for a in attentions], dim=0)
    return attn_score.mean(dim=1, keepdim=False)


def _safe_forget_num(seq_len, num_of_token, ratio):
    if num_of_token is None:
        return 0
    return int(seq_len - num_of_token * (1 - ratio))


def _select_keep_throw(score, start, end, forget_num, protect_idx, device):
    candidate_len = max(0, end - start)
    if candidate_len <= 0 or forget_num <= 0:
        selected = torch.arange(0, end, device=device)
        selected = torch.cat([selected, torch.tensor([protect_idx], device=device)], dim=0)
        throw = torch.empty(0, dtype=torch.long, device=device)
        return selected, throw

    forget_num = min(forget_num, max(candidate_len - 1, 0))
    order = torch.argsort(score[:candidate_len])
    throw = order[:forget_num] + start
    selected_middle = order[forget_num:] + start
    selected = torch.cat(
        [
            torch.arange(start, device=device),
            selected_middle,
            torch.tensor([protect_idx], device=device),
        ],
        dim=0,
    )
    return selected.long(), throw.long()


def _nearest_merge_idx(throw_idx, selected_idx):
    if throw_idx.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=selected_idx.device)
    if selected_idx.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=throw_idx.device)
    merge = []
    for t in throw_idx:
        merge.append(selected_idx[torch.abs(selected_idx - t).argmin()].unsqueeze(0))
    return torch.cat(merge).long()


class LocalCache:
    def __init__(self, start_size=4, recent_size=512, k_seq_dim=2, v_seq_dim=2, ratio=0.0):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.ratio = ratio

    def reset(self):
        pass

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        forget_num = _safe_forget_num(seq_len, num_of_token, self.ratio)
        if forget_num <= 0:
            return past_key_values
        return [
            [
                torch.cat([self.k_slice(k, 0, self.start_size), self.k_slice(k, forget_num + self.start_size, seq_len)], dim=self.k_seq_dim),
                torch.cat([self.v_slice(v, 0, self.start_size), self.v_slice(v, forget_num + self.start_size, seq_len)], dim=self.v_seq_dim),
            ]
            for k, v in past_key_values
        ]


class H2OCache:
    def __init__(self, start_size=4, recent_size=512, k_seq_dim=2, v_seq_dim=2, ratio=0.0):
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

    def reset(self):
        self.score_sum.zero_()
        self.flag = True

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        past_key_values_new = _as_legacy_past(past_key_values)
        seq_len = past_key_values_new[0][0].size(self.k_seq_dim)
        self.score_sum = _expand_scores(self.score_sum, seq_len + 1)

        attn_score = _mean_attention_per_layer(attentions).mean(dim=0, keepdim=False)
        if attn_score.shape[-2] > 1:
            if self.flag is not True:
                return past_key_values_new
            for idx in range(attn_score.shape[-2]):
                cur_score = attn_score[idx, : idx + 1]
                self.score_sum[: cur_score.shape[-1]] += cur_score
        else:
            self.score_sum[:seq_len] += attn_score.squeeze(0)

        forget_num = _safe_forget_num(seq_len, num_of_token, self.ratio)
        if forget_num <= 0:
            return past_key_values_new

        if forget_num > 1:
            if self.flag is not True:
                return past_key_values_new
            self.flag = False
            end = seq_len - self.protect_size
            selected_idx, throw_idx = _select_keep_throw(
                self.score_sum[:end], 0, end, forget_num, seq_len - self.protect_size, self.score_sum.device
            )
            out = []
            for k, v in past_key_values_new:
                out.append([_gather_seq(k, selected_idx, self.k_seq_dim), _gather_seq(v, selected_idx, self.v_seq_dim)])
            self.score_sum[: selected_idx.numel()] = self.score_sum[selected_idx].clone()
            return out

        selected_idx = self.score_sum[self.start_size : seq_len - self.protect_size].argmin() + self.start_size
        selected_idx = int(selected_idx.item())
        self.score_sum[selected_idx:-1] = self.score_sum[selected_idx + 1 :].clone()
        return [[_cat_remove(k, selected_idx, self.k_seq_dim), _cat_remove(v, selected_idx, self.v_seq_dim)] for k, v in past_key_values_new]


class ElasticCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.0,
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

    def reset(self):
        self.score_sum.zero_()
        self.flag = True
        self.selected_idx = 0

    def _update_prefill_scores(self, attn_score):
        for idx in range(attn_score.shape[-2]):
            cur_score = attn_score[:, idx, : idx + 1]
            self.score_sum[:, : cur_score.shape[-1]] += cur_score

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        self.score_sum = _expand_scores(self.score_sum, seq_len + 1)
        attn_score = _mean_attention_per_layer(attentions)

        if attn_score.shape[-2] > 1:
            if self.flag is not True:
                return past_key_values
            self._update_prefill_scores(attn_score)

        forget_num = _safe_forget_num(seq_len, num_of_token, self.ratio)
        if forget_num <= 0:
            return past_key_values

        if forget_num > 1:
            if self.flag is not True:
                return past_key_values
            self.flag = False
            selected_idx_all, merge_idx_all, throw_idx_all = [], [], []
            start = self.start_size
            end = seq_len - self.protect_size
            for layer_idx in range(self.layer_num):
                selected_idx, throw_idx = _select_keep_throw(
                    self.score_sum[layer_idx, start:end], start, end, forget_num, seq_len - self.protect_size, self.score_sum.device
                )
                selected_middle = selected_idx[self.start_size : -1]
                merge_idx = _nearest_merge_idx(throw_idx, selected_middle)
                selected_idx_all.append(selected_idx)
                merge_idx_all.append(merge_idx)
                throw_idx_all.append(throw_idx)

            self.selected_idx = self.distance if self.distance > 0 else seq_len - forget_num + self.distance
            out = []
            for layer_idx, (k, v) in enumerate(past_key_values):
                selected_idx = selected_idx_all[layer_idx]
                merge_idx = merge_idx_all[layer_idx]
                throw_idx = throw_idx_all[layer_idx]
                if throw_idx.numel() > 0 and merge_idx.numel() > 0:
                    k_forget = _gather_seq(k, throw_idx, self.k_seq_dim)
                    v_forget = _gather_seq(v, throw_idx, self.v_seq_dim)
                    k = _scatter_reduce_seq(k, merge_idx, k_forget, self.k_seq_dim, "mean")
                    v = _scatter_reduce_seq(v, merge_idx, v_forget, self.v_seq_dim, "mean")
                out.append([_gather_seq(k, selected_idx, self.k_seq_dim), _gather_seq(v, selected_idx, self.v_seq_dim)])
            return out

        selected_idx = int(self.selected_idx)
        return [[_cat_remove(k, selected_idx, self.k_seq_dim), _cat_remove(v, selected_idx, self.v_seq_dim)] for k, v in past_key_values]


class HelasticCache(ElasticCache):
    def __init__(self, *args, decay=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay = decay

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        self.score_sum = _expand_scores(self.score_sum, seq_len + 1)
        attn_score = _mean_attention_per_layer(attentions)

        if attn_score.shape[-2] > 1:
            if self.flag is not True:
                return past_key_values
            self._update_prefill_scores(attn_score)
        else:
            self.score_sum[:, :seq_len] = self.score_sum[:, :seq_len] * self.decay + attn_score.squeeze(1)

        target_cache_size = int(num_of_token * (1 - self.ratio)) if num_of_token is not None else seq_len
        forget_num = seq_len - target_cache_size
        if forget_num <= 0:
            return past_key_values

        selected_idx_all, merge_idx_all, throw_idx_all = [], [], []
        start = self.start_size
        end = seq_len - self.protect_size
        for layer_idx in range(self.layer_num):
            selected_idx, throw_idx = _select_keep_throw(
                self.score_sum[layer_idx, start:end], start, end, forget_num, seq_len - self.protect_size, self.score_sum.device
            )
            selected_middle = selected_idx[self.start_size : -1]
            merge_idx = _nearest_merge_idx(throw_idx, selected_middle)
            selected_idx_all.append(selected_idx)
            merge_idx_all.append(merge_idx)
            throw_idx_all.append(throw_idx)

        out = []
        for layer_idx, (k, v) in enumerate(past_key_values):
            selected_idx = selected_idx_all[layer_idx]
            merge_idx = merge_idx_all[layer_idx]
            throw_idx = throw_idx_all[layer_idx]
            if throw_idx.numel() > 0 and merge_idx.numel() > 0:
                k_forget = _gather_seq(k, throw_idx, self.k_seq_dim)
                v_forget = _gather_seq(v, throw_idx, self.v_seq_dim)
                k = _scatter_reduce_seq(k, merge_idx, k_forget, self.k_seq_dim, "mean")
                v = _scatter_reduce_seq(v, merge_idx, v_forget, self.v_seq_dim, "mean")
            out.append([_gather_seq(k, selected_idx, self.k_seq_dim), _gather_seq(v, selected_idx, self.v_seq_dim)])

        new_len = out[0][0].size(self.k_seq_dim)
        new_score_sum = torch.zeros(self.layer_num, new_len + 1, device=self.score_sum.device, dtype=self.score_sum.dtype)
        for layer_idx in range(self.layer_num):
            selected_idx = selected_idx_all[layer_idx]
            new_score_sum[layer_idx, : selected_idx.numel()] = self.score_sum[layer_idx, selected_idx]
        self.score_sum = new_score_sum
        self.flag = False
        return out


class MultimodalElasticCache(ElasticCache):
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
        text_image_weight=0.8,
        image_text_weight=0.8,
        image_start=None,
        image_token_len=576,
    ):
        super().__init__(start_size, recent_size, k_seq_dim, v_seq_dim, ratio, distance, layer_num)
        self.text_image_score_sum = torch.zeros(layer_num, self.cache_size + 1, device=_device())
        self.image_text_score_sum = torch.zeros(layer_num, self.cache_size + 1, device=_device())
        self.attn_weight = attn_weight
        self.text_image_weight = text_image_weight
        self.image_text_weight = image_text_weight
        self.image_start = image_start
        self.image_token_len = image_token_len

    def reset(self):
        super().reset()
        self.text_image_score_sum.zero_()
        self.image_text_score_sum.zero_()

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

    def _nearest_same_modality(self, throw_idx, selected_idx, text_mask, image_mask):
        if throw_idx.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=selected_idx.device)
        merge = []
        for t in throw_idx:
            same_selected = selected_idx[image_mask[selected_idx]] if image_mask[t].item() else selected_idx[text_mask[selected_idx]]
            if same_selected.numel() == 0:
                same_selected = selected_idx
            merge.append(same_selected[torch.abs(same_selected - t).argmin()].unsqueeze(0))
        return torch.cat(merge).long()

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        self.score_sum = _expand_scores(self.score_sum, seq_len + 1)
        self.text_image_score_sum = _expand_scores(self.text_image_score_sum, seq_len + 1)
        self.image_text_score_sum = _expand_scores(self.image_text_score_sum, seq_len + 1)

        attn_score = _mean_attention_per_layer(attentions)
        text_mask, image_mask = self._build_modality_masks(seq_len, attn_score.device)

        if attn_score.shape[-2] > 1:
            if self.flag is not True:
                return past_key_values
            for idx in range(attn_score.shape[-2]):
                cur_score = attn_score[:, idx, : idx + 1]
                self.score_sum[:, : cur_score.shape[-1]] += cur_score
                if text_mask[idx].item():
                    self.text_image_score_sum[:, : cur_score.shape[-1]] += cur_score * image_mask[: idx + 1].unsqueeze(0).float()
                else:
                    self.image_text_score_sum[:, : cur_score.shape[-1]] += cur_score * text_mask[: idx + 1].unsqueeze(0).float()

        forget_num = _safe_forget_num(seq_len, num_of_token, self.ratio)
        if forget_num <= 0:
            return past_key_values
        if forget_num > 1:
            if self.flag is not True:
                return past_key_values
            self.flag = False
            selected_idx_all, merge_idx_all, throw_idx_all = [], [], []
            start = self.start_size
            end = seq_len - self.protect_size
            for layer_idx in range(self.layer_num):
                raw_score = self.score_sum[layer_idx, start:end]
                ti_score = self.text_image_score_sum[layer_idx, start:end]
                it_score = self.image_text_score_sum[layer_idx, start:end]
                local_text_mask = text_mask[start:end]
                local_image_mask = image_mask[start:end]
                raw_norm = raw_score / (raw_score.max() + 1e-6) if raw_score.numel() else raw_score
                ti_norm = torch.zeros_like(ti_score)
                it_norm = torch.zeros_like(it_score)
                if local_image_mask.any():
                    vals = ti_score[local_image_mask]
                    ti_norm[local_image_mask] = vals / (vals.max() + 1e-6)
                if local_text_mask.any():
                    vals = it_score[local_text_mask]
                    it_norm[local_text_mask] = vals / (vals.max() + 1e-6)
                combined = self.attn_weight * raw_norm
                combined[local_image_mask] += self.text_image_weight * ti_norm[local_image_mask]
                combined[local_text_mask] += self.image_text_weight * it_norm[local_text_mask]
                selected_idx, throw_idx = _select_keep_throw(combined, start, end, forget_num, seq_len - self.protect_size, combined.device)
                selected_middle = selected_idx[self.start_size : -1]
                merge_idx = self._nearest_same_modality(throw_idx, selected_middle, text_mask, image_mask)
                selected_idx_all.append(selected_idx)
                merge_idx_all.append(merge_idx)
                throw_idx_all.append(throw_idx)

            self.selected_idx = self.distance if self.distance > 0 else seq_len - forget_num + self.distance
            out = []
            for layer_idx, (k, v) in enumerate(past_key_values):
                selected_idx = selected_idx_all[layer_idx]
                merge_idx = merge_idx_all[layer_idx]
                throw_idx = throw_idx_all[layer_idx]
                if throw_idx.numel() > 0 and merge_idx.numel() > 0:
                    k_forget = _gather_seq(k, throw_idx, self.k_seq_dim)
                    v_forget = _gather_seq(v, throw_idx, self.v_seq_dim)
                    k = _scatter_reduce_seq(k, merge_idx, k_forget, self.k_seq_dim, "mean")
                    v = _scatter_reduce_seq(v, merge_idx, v_forget, self.v_seq_dim, "mean")
                out.append([_gather_seq(k, selected_idx, self.k_seq_dim), _gather_seq(v, selected_idx, self.v_seq_dim)])
            return out

        selected_idx = int(self.selected_idx)
        return [[_cat_remove(k, selected_idx, self.k_seq_dim), _cat_remove(v, selected_idx, self.v_seq_dim)] for k, v in past_key_values]


class MultimodalHelasticCache(HelasticCache):
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
        decay=0.95,
    ):
        super().__init__(start_size, recent_size, k_seq_dim, v_seq_dim, ratio, distance, layer_num, decay=decay)
        self.conn_score_sum = torch.zeros(layer_num, self.cache_size + 1, device=_device())
        self.attn_weight = attn_weight
        self.conn_weight = conn_weight
        self.image_start = image_start
        self.image_token_len = image_token_len

    def reset(self):
        super().reset()
        self.conn_score_sum.zero_()

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

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        self.score_sum = _expand_scores(self.score_sum, seq_len + 1)
        self.conn_score_sum = _expand_scores(self.conn_score_sum, seq_len + 1)
        attn_score = _mean_attention_per_layer(attentions)
        text_mask, image_mask = self._build_modality_masks(seq_len, attn_score.device)

        if attn_score.shape[-2] > 1:
            if self.flag is not True:
                return past_key_values
            for idx in range(attn_score.shape[-2]):
                cur_score = attn_score[:, idx, : idx + 1]
                self.score_sum[:, : cur_score.shape[-1]] += cur_score
                if text_mask[idx].item():
                    conn_score = cur_score * image_mask[: idx + 1].unsqueeze(0).float()
                else:
                    conn_score = cur_score * text_mask[: idx + 1].unsqueeze(0).float()
                self.conn_score_sum[:, : cur_score.shape[-1]] += conn_score
        else:
            self.score_sum[:, :seq_len] = self.score_sum[:, :seq_len] * self.decay + attn_score.squeeze(1)

        target_cache_size = int(num_of_token * (1 - self.ratio)) if num_of_token is not None else seq_len
        forget_num = seq_len - target_cache_size
        if forget_num <= 0:
            return past_key_values

        selected_idx_all, merge_idx_all, throw_idx_all = [], [], []
        start = self.start_size
        end = seq_len - self.protect_size
        for layer_idx in range(self.layer_num):
            raw_score = self.score_sum[layer_idx, start:end]
            conn_score = self.conn_score_sum[layer_idx, start:end]
            local_text_mask = text_mask[start:end]
            raw_norm = raw_score / (raw_score.max() + 1e-6) if raw_score.numel() else raw_score
            conn_norm = torch.zeros_like(conn_score)
            if local_text_mask.any():
                vals = conn_score[local_text_mask]
                conn_norm[local_text_mask] = vals / (vals.max() + 1e-6)
            combined = self.attn_weight * raw_norm
            combined[local_text_mask] += self.conn_weight * conn_norm[local_text_mask]
            selected_idx, throw_idx = _select_keep_throw(combined, start, end, forget_num, seq_len - self.protect_size, combined.device)
            selected_middle = selected_idx[self.start_size : -1]
            merge_idx = _nearest_merge_idx(throw_idx, selected_middle)
            selected_idx_all.append(selected_idx)
            merge_idx_all.append(merge_idx)
            throw_idx_all.append(throw_idx)

        out = []
        for layer_idx, (k, v) in enumerate(past_key_values):
            selected_idx = selected_idx_all[layer_idx]
            merge_idx = merge_idx_all[layer_idx]
            throw_idx = throw_idx_all[layer_idx]
            if throw_idx.numel() > 0 and merge_idx.numel() > 0:
                k_forget = _gather_seq(k, throw_idx, self.k_seq_dim)
                v_forget = _gather_seq(v, throw_idx, self.v_seq_dim)
                k = _scatter_reduce_seq(k, merge_idx, k_forget, self.k_seq_dim, "mean")
                v = _scatter_reduce_seq(v, merge_idx, v_forget, self.v_seq_dim, "mean")
            out.append([_gather_seq(k, selected_idx, self.k_seq_dim), _gather_seq(v, selected_idx, self.v_seq_dim)])

        new_len = out[0][0].size(self.k_seq_dim)
        new_score_sum = torch.zeros(self.layer_num, new_len + 1, device=self.score_sum.device, dtype=self.score_sum.dtype)
        new_conn_score_sum = torch.zeros(self.layer_num, new_len + 1, device=self.conn_score_sum.device, dtype=self.conn_score_sum.dtype)
        for layer_idx in range(self.layer_num):
            selected_idx = selected_idx_all[layer_idx]
            new_score_sum[layer_idx, : selected_idx.numel()] = self.score_sum[layer_idx, selected_idx]
            new_conn_score_sum[layer_idx, : selected_idx.numel()] = self.conn_score_sum[layer_idx, selected_idx]
        self.score_sum = new_score_sum
        self.conn_score_sum = new_conn_score_sum
        self.flag = False
        return out
