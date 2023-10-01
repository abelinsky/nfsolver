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

from src.models.gnnprocessor import GNNProcessor
from src.models.gnnsvonv import GNNSConv
from src.models.balanceconv import BalanceConv


class GNNStableProcessor(GNNProcessor):
    """Обновляет латентные представления вершин графа за счет использования сверточной графовой нейронной сети. Возвращает расширенный список небалансов после каждого сверточного слоя."""

    def __init__(self, device="cpu", *args, **kwargs) -> None:
        super(GNNStableProcessor, self).__init__(device=device, *args, **kwargs)
        self.device = device
        self.decoders = torch.nn.ModuleList(
            [
                Linear(self.latent_dim, self.out_channels).to(device)
                for _ in range(self.num_convs)
            ]
        )

    def forward(self, data: Data):
        X = torch.zeros((data.num_nodes, self.latent_dim)).to(self.device)
        edge_index, node_attr, edge_attr = data.edge_index, data.x, data.edge_attr

        flows_list, imbalance_list = [], []

        def append_results(P_l):
            _pset = torch.where(
                data.x[..., -1] != 0, data.x[..., -1], P_l.view(-1)
            ).view(-1, 1)
            flows, imbalance = self.balance_conv(
                P=_pset, edge_index=edge_index, edge_attr=edge_attr, node_attr=node_attr
            )
            flows_list.append(flows)
            imbalance_list.append(imbalance)

        for conv, decoder in zip(self.convs, self.decoders):
            X = conv(X, edge_index, node_attr, edge_attr)
            X = F.leaky_relu(X)
            Pl = decoder(X)
            Pl = F.relu(Pl)
            append_results(Pl)

        P = self.final(X)
        P = F.relu(P)
        append_results(P)

        # P_ = torch.where(data.x[..., -1] != 0, data.x[..., -1], P.view(-1)).view(-1, 1)

        # flows, imbalance = self.balance_conv(
        #     P=P_, edge_index=edge_index, edge_attr=edge_attr, node_attr=node_attr
        # )

        # return P, flows, imbalance
        # print(f"{imbalance_list[0]=}")
        return P, torch.stack(flows_list), torch.stack(imbalance_list)
