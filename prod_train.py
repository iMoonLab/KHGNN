import logging
import os
from copy import deepcopy

# os.environ['CUDA_VISIBLE_DEIVCES']="4"
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dhg import Graph, Hypergraph
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from dhg.models import GAT, GCN, HGNN, HGNNP, HNHN, UniGAT, UniGCN
from dhg.random import set_seed
from dhg.utils import split_by_num
from omegaconf import DictConfig, OmegaConf

from khgnn_model import KerHGNN
from utils import fix_iso_v, load_data, product_split, re_index, sub_hypergraph


def train(net, X, G, lbls, train_mask, optimizer):
    net.train()
    optimizer.zero_grad()
    outs = net(X, G)
    # loss=F.nll_loss(F.softmax(outs[train_mask],dim=1),lbls[train_mask])
    loss = F.cross_entropy(outs[train_mask], lbls[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def valid(net, X, G, lbls, mask, evaluator: Evaluator = None):
    net.eval()
    outs = net(X, G)
    res = evaluator.validate(lbls[mask], outs[mask])
    return res


def test(
    net,
    X_t,
    G_t,
    X,
    G,
    lbls_t,
    lbls,
    evaluator: Evaluator,
    mask_t=None,
    mask_i=None,
    mask_p=None,
):
    net.eval()
    # transductive
    outs = net(X_t, G_t)
    res_t = evaluator.test(lbls_t[mask_t], outs[mask_t])
    # inductive
    outs = net(X, G)
    res_i = evaluator.test(lbls[mask_i], outs[mask_i])
    # product
    outs = net(X, G)
    res_p = evaluator.test(lbls[mask_p], outs[mask_p])
    res = {}
    for k, v in res_t.items():
        res[f"trans_{k}"] = v
    for k, v in res_i.items():
        res[f"ind_{k}"] = v
    for k, v in res_p.items():
        res[f"prod_{k}"] = v
    return res

    return res


# ==================================== #
def exp(seed, cfg: DictConfig):
    set_seed(seed)
    device = (
        torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")
    )
    # device=torch.device('cpu')
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    # ============data===================#
    data, edge_list = load_data(cfg.data.name)
    if cfg.model.name in ["gcn", "gat", "pna", "gnnagg"]:  # graph model
        if cfg.data.name in ["cora", "pubmed", "citeseer"]:
            G = Graph(data["num_vertices"], edge_list)
        else:
            g = Hypergraph(data["num_vertices"], edge_list)
            G = Graph.from_hypergraph_clique(g)
        G.add_extra_selfloop()
    else:  # hypergraph model
        if cfg.data.name in ["cora", "pubmed", "citeseer"]:
            g = Graph(data["num_vertices"], edge_list)
            G = Hypergraph.from_graph(g)
            G.add_hyperedges_from_graph_kHop(g, 1)
        else:
            G = Hypergraph(data["num_vertices"], edge_list)
        if cfg.data.self_loop is True:
            G = fix_iso_v(G)
    if cfg.data.random_split is True:
        train_mask, val_mask, test_mask = split_by_num(
            data["num_vertices"], data["labels"], cfg.data.num_train, cfg.data.num_val
        )
    else:
        train_mask, val_mask, test_mask = (
            data["train_mask"],
            data["val_mask"],
            data["test_mask"],
        )
    obs_idx, obs_train_mask, obs_val_mask, obs_test_mask, test_ind_mask = product_split(
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        test_ind_ratio=cfg.data.test_ind_ratio,
    )
    G_t = sub_hypergraph(G, obs_idx)
    G_t = fix_iso_v(G_t)
    # ================================================#
    try:
        X = data["features"]
    except:
        X = torch.eye(data["num_vertices"])
        # data['dim_features']=data['num_vertices']
    lbls = data["labels"]
    X_t, lbls_t = X[obs_idx], lbls[obs_idx]
    num_v, num_e = G.num_v, G.num_e
    # ====================model========================#
    if cfg.model.name == "hgnn":
        net = HGNN(X.shape[1], 32, data["num_classes"], use_bn=False)
    elif cfg.model.name == "hgnnp":
        net = HGNNP(X.shape[1], 32, data["num_classes"], use_bn=False)
    elif cfg.model.name == "unignn":
        net = UniGCN(X.shape[1], 32, data["num_classes"], use_bn=False)
    elif cfg.model.name == "unigat":
        net = UniGAT(X.shape[1], 8, data["num_classes"], use_bn=False, num_heads=4)
    elif cfg.model.name == "hnhn":
        net = HNHN(X.shape[1], 32, data["num_classes"], use_bn=False)
    elif cfg.model.name == "gcn":
        net = GCN(X.shape[1], 32, data["num_classes"], use_bn=False)
    elif cfg.model.name == "gat":
        net = GAT(X.shape[1], 8, data["num_classes"], use_bn=False, num_heads=4)
    elif cfg.model.name == "kerhgnn":
        net = KerHGNN(
            X.shape[1],
            cfg.model.hid,
            data["num_classes"],
            use_bn=False,
            drop_rate=0.5,
            num_layer=cfg.model.num_layer,
            kernel_type=cfg.model.kernel_type,
            p_min=cfg.model.p_min,
            p_max=cfg.model.p_max,
            mu=cfg.model.mu,
        )
    else:
        raise NotImplementedError
    # ================train valid test==================#

    if cfg.model.name == "kerhgnn":
        model_named_parameters = [net.layers[i].p for i in range(cfg.model.num_layer)]
        hgnnconv = list(map(id, model_named_parameters))
        based_params = filter(lambda p: id(p) not in hgnnconv, net.parameters())
        optimizer = torch.optim.Adam(
            [
                {"params": net.layers[0].p, "lr": cfg.optim.lr_p},
                {"params": net.layers[1].p, "lr": cfg.optim.lr_p},
                {"params": based_params, "lr": cfg.optim.lr},
            ],
            weight_decay=5e-4,
        )
    else:
        optimizer = optim.Adam(net.parameters(), lr=cfg.optim.lr_p, weight_decay=5e-4)
    net = net.to(device)
    X, lbls, G = X.to(device), lbls.to(device), G.to(device)
    X_t, lbls_t, G_t = X_t.to(device), lbls_t.to(device), G_t.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        loss = train(net, X_t, G_t, lbls_t, obs_train_mask, optimizer)
        # print('loss:',loss)
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = valid(net, X_t, G_t, lbls_t, obs_val_mask, evaluator)
                # print('loss:',loss,'val_res:',val_res)
            if val_res > best_val:
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())

    net.load_state_dict(best_state)
    res = test(
        net=net,
        X_t=X_t,
        G_t=G_t,
        X=X,
        G=G,
        lbls_t=lbls_t,
        lbls=lbls,
        evaluator=evaluator,
        mask_t=obs_test_mask,
        mask_i=test_ind_mask,
        mask_p=test_mask,
    )
    logging.info(f"test best epoch:{best_epoch},res {res}")
    return res


@hydra.main(config_path=".", config_name="prod_config", version_base="1.2")
def main(cfg: DictConfig):
    res = exp(2023, cfg)
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info(f"test:{res}")


if __name__ == "__main__":
    main()
