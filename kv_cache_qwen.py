import torch


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


def _make_index(idx, x, seq_dim):
    """Build gather/scatter index for a 4-D KV tensor and arbitrary sequence dim."""
    idx = idx.to(x.device).long()
    shape = [1] * x.dim()
    shape[seq_dim] = -1
    expand_shape = list(x.shape)
    expand_shape[seq_dim] = -1
    return idx.view(*shape).expand(*expand_shape)


def _gather_seq(x, idx, seq_dim):
    return x.gather(dim=seq_dim, index=_make_index(idx, x, seq_dim))


def _scatter_reduce_seq(x, idx, src, seq_dim):
    return x.scatter_reduce(
        dim=seq_dim,
        index=_make_index(idx, x, seq_dim),
        src=src,
        reduce="mean",
    )


def _cat_keep(k, selected_idx, seq_dim):
    return _gather_seq(k, selected_idx, seq_dim)


def _safe_cat(items):
    if len(items) == 0:
        return None
    return torch.cat(items)


class ElasticCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=1,
        v_seq_dim=1,
        ratio=0.0,
        distance=-25,
        layer_num=32,
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

    def _expand_score(self, seq_len):
        if seq_len > self.score_sum.shape[-1]:
            new_score = torch.zeros(self.layer_num, seq_len + 1, device=self.score_sum.device)
            new_score[:, : self.score_sum.shape[-1]] = self.score_sum
            self.score_sum = new_score

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        if attentions is None or num_of_token is None:
            return past_key_values

        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        self._expand_score(seq_len)

        attn_score = torch.cat([a for a in attentions], dim=0).mean(dim=1, keepdim=False)

        if attn_score.shape[-2] > 1:
            if self.flag is not True:
                return past_key_values
            for idx in range(attn_score.shape[-2]):
                cur_score = attn_score[:, idx, : idx + 1]
                self.score_sum[:, : cur_score.shape[-1]] += cur_score.to(self.score_sum.device)

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
                if end <= start:
                    return past_key_values

                scores = self.score_sum[layer_idx, start:end]
                order = torch.argsort(scores)
                keep_local = order[forget_num:]
                throw_local = order[:forget_num]

                selected_idx = keep_local + start
                throw_idx = throw_local + start

                if selected_idx.numel() == 0 or throw_idx.numel() == 0:
                    return past_key_values

                merge_idx = []
                for t in throw_idx:
                    merge_idx.append(selected_idx[torch.abs(selected_idx - t).argmin()].unsqueeze(0))
                merge_idx = torch.cat(merge_idx)

                selected_idx = torch.cat(
                    [
                        torch.arange(self.start_size, device=self.score_sum.device),
                        selected_idx,
                        torch.tensor([seq_len - self.protect_size], device=self.score_sum.device),
                    ],
                    dim=0,
                )

                selected_idx_all.append(selected_idx)
                merge_idx_all.append(merge_idx)
                throw_idx_all.append(throw_idx)

            self.selected_idx = self.distance if self.distance > 0 else seq_len - forget_num + self.distance

            past_key_values_return = []
            for layer_idx, (k, v) in enumerate(past_key_values):
                selected_idx = selected_idx_all[layer_idx].to(k.device)
                merge_idx = merge_idx_all[layer_idx].to(k.device)
                throw_idx = throw_idx_all[layer_idx].to(k.device)

                k_forget = _gather_seq(k, throw_idx, self.k_seq_dim)
                v_forget = _gather_seq(v, throw_idx, self.v_seq_dim)

                k = _scatter_reduce_seq(k, merge_idx, k_forget, self.k_seq_dim)
                v = _scatter_reduce_seq(v, merge_idx, v_forget, self.v_seq_dim)

                k_new = _gather_seq(k, selected_idx, self.k_seq_dim)
                v_new = _gather_seq(v, selected_idx, self.v_seq_dim)
                past_key_values_return.append([k_new, v_new])

            return past_key_values_return

        selected_idx = self.selected_idx
        return [
            [
                torch.cat([self.k_slice(k, 0, selected_idx), self.k_slice(k, selected_idx + 1, seq_len)], dim=self.k_seq_dim),
                torch.cat([self.v_slice(v, 0, selected_idx), self.v_slice(v, selected_idx + 1, seq_len)], dim=self.v_seq_dim),
            ]
            for k, v in past_key_values
        ]


class HelasticCache(ElasticCache):
    """Elastic cache with H2O-style dynamic score update during decoding."""

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        if attentions is None or num_of_token is None:
            return past_key_values

        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        self._expand_score(seq_len)

        attn_score = torch.cat([a for a in attentions], dim=0).mean(dim=1, keepdim=False)

        if attn_score.shape[-2] > 1:
            if self.flag is not True:
                return past_key_values
            for idx in range(attn_score.shape[-2]):
                cur_score = attn_score[:, idx, : idx + 1]
                self.score_sum[:, : cur_score.shape[-1]] += cur_score.to(self.score_sum.device)
        else:
            self.score_sum[:, :seq_len] = (self.score_sum[:, :seq_len] * 0.95) + attn_score.squeeze(1).to(self.score_sum.device)

        target_cache_size = int(num_of_token * (1 - self.ratio))
        forget_num = seq_len - target_cache_size
        if forget_num <= 0:
            return past_key_values

        start = self.start_size
        end = seq_len - self.protect_size
        if end <= start:
            return past_key_values

        selected_idx_all = []
        merge_idx_all = []
        throw_idx_all = []

        for layer_idx in range(self.layer_num):
            scores = self.score_sum[layer_idx, start:end]
            order = torch.argsort(scores)
            throw_local = order[:forget_num]
            selected_local = order[forget_num:]

            selected_idx = selected_local + start
            throw_idx = throw_local + start
            if selected_idx.numel() == 0 or throw_idx.numel() == 0:
                return past_key_values

            merge_idx = []
            for t in throw_idx:
                merge_idx.append(selected_idx[torch.abs(selected_idx - t).argmin()].unsqueeze(0))
            merge_idx = torch.cat(merge_idx)

            selected_idx = torch.cat(
                [
                    torch.arange(self.start_size, device=self.score_sum.device),
                    selected_idx,
                    torch.tensor([seq_len - self.protect_size], device=self.score_sum.device),
                ],
                dim=0,
            )

            selected_idx_all.append(selected_idx)
            merge_idx_all.append(merge_idx)
            throw_idx_all.append(throw_idx)

        past_key_values_return = []
        for layer_idx, (k, v) in enumerate(past_key_values):
            selected_idx = selected_idx_all[layer_idx].to(k.device)
            merge_idx = merge_idx_all[layer_idx].to(k.device)
            throw_idx = throw_idx_all[layer_idx].to(k.device)

            k_forget = _gather_seq(k, throw_idx, self.k_seq_dim)
            v_forget = _gather_seq(v, throw_idx, self.v_seq_dim)
            k = _scatter_reduce_seq(k, merge_idx, k_forget, self.k_seq_dim)
            v = _scatter_reduce_seq(v, merge_idx, v_forget, self.v_seq_dim)
            k_new = _gather_seq(k, selected_idx, self.k_seq_dim)
            v_new = _gather_seq(v, selected_idx, self.v_seq_dim)
            past_key_values_return.append([k_new, v_new])

        new_score_sum = torch.zeros(self.layer_num, target_cache_size + 1, device=self.score_sum.device)
        for layer_idx in range(self.layer_num):
            idx = selected_idx_all[layer_idx].to(self.score_sum.device)
            new_score_sum[layer_idx, : len(idx)] = self.score_sum[layer_idx, idx]
        self.score_sum = new_score_sum

        return past_key_values_return


class MultimodalElasticCache(ElasticCache):
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=1,
        v_seq_dim=1,
        ratio=0.0,
        distance=-25,
        layer_num=32,
        attn_weight=1.0,
        conn_weight=0.8,
        image_start=None,
        image_token_len=256,
    ):
        super().__init__(start_size, recent_size, k_seq_dim, v_seq_dim, ratio, distance, layer_num)
        self.conn_score_sum = torch.zeros(layer_num, self.cache_size + 1, device=self.score_sum.device)
        self.attn_weight = attn_weight
        self.conn_weight = conn_weight
        self.image_start = image_start
        self.image_token_len = image_token_len

    def reset(self):
        self.score_sum.zero_()
        self.conn_score_sum.zero_()
        self.flag = True
        self.selected_idx = 0

    def _expand_score(self, seq_len):
        super()._expand_score(seq_len)
        if seq_len > self.conn_score_sum.shape[-1]:
            new_score = torch.zeros(self.layer_num, seq_len + 1, device=self.conn_score_sum.device)
            new_score[:, : self.conn_score_sum.shape[-1]] = self.conn_score_sum
            self.conn_score_sum = new_score

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
        if attentions is None or num_of_token is None:
            return past_key_values

        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        self._expand_score(seq_len)
        attn_score = torch.cat([a for a in attentions], dim=0).mean(dim=1, keepdim=False)
        text_mask, image_mask = self._build_modality_masks(seq_len, attn_score.device)

        if attn_score.shape[-2] > 1:
            if self.flag is not True:
                return past_key_values
            for idx in range(attn_score.shape[-2]):
                cur_score = attn_score[:, idx, : idx + 1]
                cur_score_dev = cur_score.to(self.score_sum.device)
                self.score_sum[:, : cur_score.shape[-1]] += cur_score_dev

                if text_mask[idx].item():
                    conn_score = cur_score * image_mask[: idx + 1].unsqueeze(0).float()
                else:
                    conn_score = cur_score * text_mask[: idx + 1].unsqueeze(0).float()
                self.conn_score_sum[:, : conn_score.shape[-1]] += conn_score.to(self.conn_score_sum.device)

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
            start = self.start_size
            end = seq_len - self.protect_size
            if end <= start:
                return past_key_values

            text_mask = text_mask.to(self.score_sum.device)
            for layer_idx in range(self.layer_num):
                raw_score = self.score_sum[layer_idx, start:end]
                conn_score = self.conn_score_sum[layer_idx, start:end]
                local_text_mask = text_mask[start:end]

                raw_score_norm = raw_score / (raw_score.max() + 1e-6)
                conn_score_norm = torch.zeros_like(conn_score)
                if local_text_mask.any():
                    conn_text = conn_score[local_text_mask]
                    conn_score_norm[local_text_mask] = conn_text / (conn_text.max() + 1e-6)

                combined_score = self.attn_weight * raw_score_norm
                combined_score[local_text_mask] += self.conn_weight * conn_score_norm[local_text_mask]

                order = torch.argsort(combined_score)
                throw_idx = order[:forget_num] + start
                selected_idx = order[forget_num:] + start
                if selected_idx.numel() == 0 or throw_idx.numel() == 0:
                    return past_key_values

                merge_idx = []
                for t in throw_idx:
                    merge_idx.append(selected_idx[torch.abs(selected_idx - t).argmin()].unsqueeze(0))
                merge_idx = torch.cat(merge_idx)

                selected_idx = torch.cat(
                    [
                        torch.arange(self.start_size, device=self.score_sum.device),
                        selected_idx,
                        torch.tensor([seq_len - self.protect_size], device=self.score_sum.device),
                    ],
                    dim=0,
                )

                selected_idx_all.append(selected_idx)
                merge_idx_all.append(merge_idx)
                throw_idx_all.append(throw_idx)

            self.selected_idx = self.distance if self.distance > 0 else seq_len - forget_num + self.distance

            past_key_values_return = []
            for layer_idx, (k, v) in enumerate(past_key_values):
                selected_idx = selected_idx_all[layer_idx].to(k.device)
                merge_idx = merge_idx_all[layer_idx].to(k.device)
                throw_idx = throw_idx_all[layer_idx].to(k.device)
                k_forget = _gather_seq(k, throw_idx, self.k_seq_dim)
                v_forget = _gather_seq(v, throw_idx, self.v_seq_dim)
                k = _scatter_reduce_seq(k, merge_idx, k_forget, self.k_seq_dim)
                v = _scatter_reduce_seq(v, merge_idx, v_forget, self.v_seq_dim)
                k_new = _gather_seq(k, selected_idx, self.k_seq_dim)
                v_new = _gather_seq(v, selected_idx, self.v_seq_dim)
                past_key_values_return.append([k_new, v_new])
            return past_key_values_return

        selected_idx = self.selected_idx
        return [
            [
                torch.cat([self.k_slice(k, 0, selected_idx), self.k_slice(k, selected_idx + 1, seq_len)], dim=self.k_seq_dim),
                torch.cat([self.v_slice(v, 0, selected_idx), self.v_slice(v, selected_idx + 1, seq_len)], dim=self.v_seq_dim),
            ]
            for k, v in past_key_values
        ]


class MultimodalHelasticCache(MultimodalElasticCache):
    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        if attentions is None or num_of_token is None:
            return past_key_values

        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        self._expand_score(seq_len)
        attn_score = torch.cat([a for a in attentions], dim=0).mean(dim=1, keepdim=False)
        text_mask, image_mask = self._build_modality_masks(seq_len, attn_score.device)

        if attn_score.shape[-2] > 1:
            if self.flag is not True:
                return past_key_values
            for idx in range(attn_score.shape[-2]):
                cur_score = attn_score[:, idx, : idx + 1]
                self.score_sum[:, : cur_score.shape[-1]] += cur_score.to(self.score_sum.device)
                if text_mask[idx].item():
                    conn_score = cur_score * image_mask[: idx + 1].unsqueeze(0).float()
                else:
                    conn_score = cur_score * text_mask[: idx + 1].unsqueeze(0).float()
                self.conn_score_sum[:, : conn_score.shape[-1]] += conn_score.to(self.conn_score_sum.device)
        else:
            self.score_sum[:, :seq_len] = (self.score_sum[:, :seq_len] * 0.95) + attn_score.squeeze(1).to(self.score_sum.device)

        target_cache_size = int(num_of_token * (1 - self.ratio))
        forget_num = seq_len - target_cache_size
        if forget_num <= 0:
            return past_key_values

        selected_idx_all = []
        merge_idx_all = []
        throw_idx_all = []
        start = self.start_size
        end = seq_len - self.protect_size
        if end <= start:
            return past_key_values

        text_mask = text_mask.to(self.score_sum.device)
        for layer_idx in range(self.layer_num):
            raw_score = self.score_sum[layer_idx, start:end]
            conn_score = self.conn_score_sum[layer_idx, start:end]
            local_text_mask = text_mask[start:end]

            raw_score_norm = raw_score / (raw_score.max() + 1e-6)
            conn_score_norm = torch.zeros_like(conn_score)
            if local_text_mask.any():
                conn_text = conn_score[local_text_mask]
                conn_score_norm[local_text_mask] = conn_text / (conn_text.max() + 1e-6)

            combined_score = self.attn_weight * raw_score_norm
            combined_score[local_text_mask] += self.conn_weight * conn_score_norm[local_text_mask]

            order = torch.argsort(combined_score)
            throw_idx = order[:forget_num] + start
            selected_idx = order[forget_num:] + start
            if selected_idx.numel() == 0 or throw_idx.numel() == 0:
                return past_key_values

            merge_idx = []
            for t in throw_idx:
                merge_idx.append(selected_idx[torch.abs(selected_idx - t).argmin()].unsqueeze(0))
            merge_idx = torch.cat(merge_idx)

            selected_idx = torch.cat(
                [
                    torch.arange(self.start_size, device=self.score_sum.device),
                    selected_idx,
                    torch.tensor([seq_len - self.protect_size], device=self.score_sum.device),
                ],
                dim=0,
            )
            selected_idx_all.append(selected_idx)
            merge_idx_all.append(merge_idx)
            throw_idx_all.append(throw_idx)

        past_key_values_return = []
        for layer_idx, (k, v) in enumerate(past_key_values):
            selected_idx = selected_idx_all[layer_idx].to(k.device)
            merge_idx = merge_idx_all[layer_idx].to(k.device)
            throw_idx = throw_idx_all[layer_idx].to(k.device)
            k_forget = _gather_seq(k, throw_idx, self.k_seq_dim)
            v_forget = _gather_seq(v, throw_idx, self.v_seq_dim)
            k = _scatter_reduce_seq(k, merge_idx, k_forget, self.k_seq_dim)
            v = _scatter_reduce_seq(v, merge_idx, v_forget, self.v_seq_dim)
            k_new = _gather_seq(k, selected_idx, self.k_seq_dim)
            v_new = _gather_seq(v, selected_idx, self.v_seq_dim)
            past_key_values_return.append([k_new, v_new])

        new_score_sum = torch.zeros(self.layer_num, target_cache_size + 1, device=self.score_sum.device)
        new_conn_score_sum = torch.zeros(self.layer_num, target_cache_size + 1, device=self.conn_score_sum.device)
        for layer_idx in range(self.layer_num):
            idx = selected_idx_all[layer_idx].to(self.score_sum.device)
            new_score_sum[layer_idx, : len(idx)] = self.score_sum[layer_idx, idx]
            new_conn_score_sum[layer_idx, : len(idx)] = self.conn_score_sum[layer_idx, idx]
        self.score_sum = new_score_sum
        self.conn_score_sum = new_conn_score_sum

        return past_key_values_return


class LocalCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=1,
        v_seq_dim=1,
        ratio=0.0,
    ):
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
        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
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
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=1,
        v_seq_dim=1,
        ratio=0.0,
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

    def reset(self):
        self.score_sum.zero_()
        self.flag = True

    def _expand_score(self, seq_len):
        if seq_len > self.score_sum.shape[-1]:
            new_score = torch.zeros(seq_len + 1, device=self.score_sum.device)
            new_score[: self.score_sum.shape[-1]] = self.score_sum
            self.score_sum = new_score

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        if attentions is None or num_of_token is None:
            return past_key_values

        past_key_values_new = tuple(x for x in past_key_values)
        seq_len = past_key_values_new[0][0].size(self.k_seq_dim)
        self._expand_score(seq_len)

        attn_score = torch.cat([a for a in attentions], dim=0).mean(dim=1, keepdim=False).mean(dim=0, keepdim=False)

        if attn_score.shape[-2] > 1:
            if self.flag is not True:
                return past_key_values_new
            for idx in range(attn_score.shape[-2]):
                cur_score = attn_score[idx][: idx + 1]
                self.score_sum[: len(cur_score)] += cur_score.to(self.score_sum.device)
        else:
            attn_score = attn_score.squeeze(0)
            self.score_sum[:seq_len] += attn_score.to(self.score_sum.device)

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        if forget_num <= 0:
            return past_key_values_new

        if forget_num > 1:
            if self.flag is not True:
                return past_key_values_new
            self.flag = False
            selected_idx = torch.argsort(self.score_sum[: seq_len - self.protect_size])[forget_num:]
            selected_idx = torch.cat(
                [selected_idx, torch.arange(seq_len - self.protect_size, seq_len, device=self.score_sum.device)],
                dim=0,
            )
            return [[_gather_seq(k, selected_idx.to(k.device), self.k_seq_dim), _gather_seq(v, selected_idx.to(v.device), self.v_seq_dim)] for k, v in past_key_values_new]

        selected_idx = self.score_sum[self.start_size : seq_len - self.protect_size].argmin() + self.start_size
        self.score_sum[selected_idx:-1] = self.score_sum[selected_idx + 1 :].clone()
        return [
            [
                torch.cat([self.k_slice(k, 0, selected_idx), self.k_slice(k, selected_idx + 1, seq_len)], dim=self.k_seq_dim),
                torch.cat([self.v_slice(v, 0, selected_idx), self.v_slice(v, selected_idx + 1, seq_len)], dim=self.v_seq_dim),
            ]
            for k, v in past_key_values_new
        ]
