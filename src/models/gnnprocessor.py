import torch
from torch.nn import Sequential, Linear, Parameter, ReLU, ELU
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, LayerNorm
from torch_geometric.data import Data
from typing import Any, Dict, List
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor

import networkx as nx
import matplotlib.pyplot as plt

from src.models.gnnsvonv import GNNSConv
from src.models.balanceconv import BalanceConv


class GNNProcessor(torch.nn.Module):
    """Обновляет латентные представления вершин графа за счет использования сверточной графовой нейронной сети."""

    def __init__(
        self,
        out_channels: int,
        num_edge_features: int,
        latent_dim: int = 6,
        num_convs: int = 8,
    ) -> None:
        """Инициализация

        Args:
            num_features (int): _description_
            num_classes (int): _description_
            num_edge_features (int): _description_
            latent_dim (int, optional): Размерность латентного пространства значений в вершинах графа (по умолчанию 10).
            num_convs (int, optional): Количество слоев свертки.
        """
        super().__init__()
        # torch.manual_seed(1234)
        self.latent_dim = latent_dim
        self.convs = torch.nn.ModuleList(
            [
                GNNSConv(
                    latent_dim=self.latent_dim,
                    num_edge_features=num_edge_features,
                    hidden_layers=[16],
                    alpha=1.0,
                )
                for _ in range(num_convs)
            ]
        )
        self.final = Linear(latent_dim, out_channels)
        self.balance_conv = BalanceConv()

    def forward(self, data: Data):
        X = torch.zeros((data.num_nodes, self.latent_dim))
        edge_index, node_attr, edge_attr = data.edge_index, data.x, data.edge_attr

        for conv in self.convs:
            X = conv(X, edge_index, node_attr, edge_attr)
            X = F.relu(X)

        P = self.final(X)
        P = F.relu(P)

        P_ = torch.where(data.x[..., -1] != 0, data.x[..., -1], P.view(-1)).view(-1, 1)

        flows, imbalance = self.balance_conv(
            P=P_, edge_index=edge_index, edge_attr=edge_attr, node_attr=node_attr
        )

        return P, flows, imbalance
