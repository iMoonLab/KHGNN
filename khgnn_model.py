import dhg
import torch
import torch.nn as nn

from utils import NodeNorm

EPS = 1e-5


class KerHGNNConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = True,
        drop_rate: float = 0.5,
        is_last: bool = False,
        kernel_type: str = "linear",
        p_min: int = 0,
        p_max: int = 2,
        mu: int = 1,
    ):
        super(KerHGNNConv, self).__init__()
        self.in_channels = in_channels
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        # nn.init.kaiming_uniform_(self.kernel.data)
        self.kernel_type = kernel_type
        self.p = nn.Parameter(torch.FloatTensor(1, 1))
        nn.init.constant_(self.p, 0.3150)
        self.p_min = p_min
        self.p_max = p_max
        self.mu = mu
        self.norm = NodeNorm()

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph):
        ####
        if self.kernel_type not in ["poly", "apoly", "mean"]:
            print(
                "wrong kernel type, please change args.kernel in [adaptive poly genMean]"
            )
            raise NotImplementedError
        X = self.theta(X)
        if self.kernel_type == "poly":
            X = self.poly_agg_v2e(X=X, hg=hg)
            X = hg.v2e_update(X)
            X = hg.e2v(X, "mean", drop_rate=0.0)
        elif self.kernel_type == "apoly":
            X = self.poly_agg_v2e(X=X, hg=hg)
            X = hg.v2e_update(X)
            X = self.poly_agg_e2v(X=X, hg=hg)
            X = hg.e2v_update(X)
        else:
            X = hg.v2v(X, "mean", drop_rate=0.0)
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X

    def poly_agg_v2e(self, X: torch.Tensor, hg: dhg.Hypergraph):
        X = self.norm(X)
        p = torch.clamp(self.p, self.p_min, self.p_max)
        min_u = torch.min(X) - self.mu - EPS
        X = (
            torch.sparse.mm(hg.H_T, torch.pow(X - min_u, p + 1)).div(
                torch.sparse.mm(hg.H_T, torch.pow(X - min_u, p)) + EPS
            )
            + min_u
        )
        return X

    def poly_agg_e2v(self, X: torch.Tensor, hg: dhg.Hypergraph):
        p = torch.clamp(self.p, self.p_min, self.p_max)
        min_u = torch.min(X) - self.mu - EPS
        X = (
            torch.sparse.mm(hg.H, torch.pow(X - min_u, p + 1)).div(
                torch.sparse.mm(hg.H, torch.pow(X - min_u, p)) + EPS
            )
            + min_u
        )
        return X


class KerHGNN(nn.Module):
    r"""
    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        kernel_type: str = "poly",
        p_min=-0.5,
        p_max=2,
        mu=1,
        num_layer: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            KerHGNNConv(
                in_channels,
                hid_channels,
                use_bn=use_bn,
                drop_rate=drop_rate,
                kernel_type=kernel_type,
                p_min=p_min,
                p_max=p_max,
                mu=mu,
            )
        )
        for _ in range(num_layer - 2):
            self.layers.append(
                KerHGNNConv(
                    in_channels=hid_channels,
                    out_channels=hid_channels,
                    drop_rate=drop_rate,
                    kernel_type=kernel_type,
                    p_min=p_min,
                    p_max=p_max,
                    mu=mu,
                )
            )
        self.layers.append(
            KerHGNNConv(
                hid_channels,
                num_classes,
                use_bn=use_bn,
                drop_rate=drop_rate,
                is_last=True,
                kernel_type=kernel_type,
                p_min=p_min,
                p_max=p_max,
                mu=mu,
            )
        )

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
        # r"""The forward function.
        # Args:
        #     ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
        #     ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        # """
        for layer in self.layers:
            X = layer(X, hg)
        return X
