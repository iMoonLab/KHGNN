from collections import defaultdict

import dhg
import numpy as np
import torch
import torch.nn as nn
from dhg.data import (
    Citeseer,
    CoauthorshipCora,
    CoauthorshipDBLP,
    CocitationCiteseer,
    CocitationCora,
    CocitationPubmed,
    Cooking200,
    Cora,
    DBLP4k,
    IMDB4k,
    News20,
    Pubmed,
)


def load_data(name):
    if name == "coauthorship_cora":
        data = CoauthorshipCora()
        edge_list = data["edge_list"]
    elif name == "coauthorship_dblp":
        data = CoauthorshipDBLP()
        edge_list = data["edge_list"]
    elif name == "cocitation_cora":
        data = CocitationCora()
        edge_list = data["edge_list"]
    elif name == "cocitation_pubmed":
        data = CocitationPubmed()
        edge_list = data["edge_list"]
    elif name == "cocitation_citeseer":
        data = CocitationCiteseer()
        edge_list = data["edge_list"]
    elif name == "news20":
        data = News20()
        edge_list = data["edge_list"]
    elif name == "dblp4k-paper":
        data = DBLP4k()
        edge_list = data["edge_by_paper"]
    elif name == "dblp4k-term":
        data = DBLP4k()
        edge_list = data["edge_by_term"]
    elif name == "dblp4k-conf":
        data = DBLP4k()
        edge_list = data["edge_by_conf"]
    elif name == "imdb4k":
        data = IMDB4k()
        edge_list = data["edge_by_actor"] + data["edge_by_director"]
    elif name == "cora":
        data = Cora()
        edge_list = data["edge_list"]
    elif name == "pubmed":
        data = Pubmed()
        edge_list = data["edge_list"]
    elif name == "citeseer":
        data = Citeseer()
        edge_list = data["edge_list"]
    elif name == "cooking200":
        data = Cooking200()
        edge_list = data["edge_list"]
    else:
        raise NotImplementedError
    return data, edge_list


def fix_iso_v(G: dhg.Hypergraph):
    iso_v = np.array(G.deg_v) == 0
    if np.any(iso_v):
        extra_e = [
            tuple(
                [
                    e,
                ]
            )
            for e in np.where(iso_v)[0]
        ]
        G.add_hyperedges(extra_e)
    return G


class MultiExpMetric:
    def __init__(self) -> None:
        self.model = defaultdict(list)

    def update(self, res):
        self._update(self.model, res)

    def _update(self, data, new_res):
        for k, v in new_res.items():
            data[k].append(v)

    def __str__(
        self,
    ):
        ret = []
        ret.append("model:")
        for k, v in self.model.items():
            v = np.array(v)
            ret.append(f"\t{k} -> {v.mean():.5f} - {v.std():.5f}")
        return "\n".join(ret)


class NodeNorm(nn.Module):
    def __init__(self, eps=1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = (torch.var(x, dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        return x


class PairNorm(nn.Module):
    def __init__(
        self, scale: float = 1, scale_individually: bool = False, eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.scale = scale
        self.scale_individually = scale_individually
        self.eps = eps

    def forward(self, X: torch.Tensor):
        scale = self.scale
        X = X - X.mean(dim=0, keepdim=True)
        if not self.scale_individually:
            return scale * X / (self.eps + X.pow(2).sum(-1).mean().sqrt())
        else:
            return scale * X / (self.eps + X.norm(2, -1, keepdim=True))


def sub_hypergraph(hg: dhg.Hypergraph, v_idx):
    v_map = {v: idx for idx, v in enumerate(v_idx)}
    v_set = set(v_idx)
    e_list, w_list = [], []
    for e, w in zip(*hg.e):
        new_e = []
        for v in e:
            if v in v_set:
                new_e.append(v_map[v])
        if len(new_e) >= 1:
            e_list.append(tuple(new_e))
            w_list.append(w)
    return dhg.Hypergraph(len(v_set), e_list, w_list)


def product_split(train_mask, val_mask, test_mask, test_ind_ratio):
    train_idx, val_idx, test_idx = (
        torch.where(train_mask)[0],
        torch.where(val_mask)[0],
        torch.where(test_mask)[0],
    )
    test_idx_shuffle = torch.randperm(len(test_idx))
    num_ind = int(len(test_idx) * test_ind_ratio)
    test_ind_idx, test_tran_idx = (
        test_idx[test_idx_shuffle[:num_ind]],
        test_idx[test_idx_shuffle[num_ind:]],
    )
    obs_idx = torch.cat([train_idx, val_idx, test_tran_idx]).numpy().tolist()

    num_obs, num_train, num_val = len(obs_idx), len(train_idx), len(val_idx)
    test_ind_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    obs_train_mask = torch.zeros(num_obs, dtype=torch.bool)
    obs_val_mask = torch.zeros(num_obs, dtype=torch.bool)
    obs_test_mask = torch.zeros(num_obs, dtype=torch.bool)

    test_ind_mask[test_ind_idx] = True
    obs_train_mask[:num_train] = True
    obs_val_mask[num_train : num_train + num_val] = True
    obs_test_mask[num_train + num_val :] = True
    return obs_idx, obs_train_mask, obs_val_mask, obs_test_mask, test_ind_mask


def re_index(vec):
    res = vec.clone()
    raw_id, new_id = res[0].item(), 0
    for idx in range(len(vec)):
        if vec[idx].item() != raw_id:
            raw_id, new_id = vec[idx].item(), new_id + 1
        res[idx] = new_id
    return res
