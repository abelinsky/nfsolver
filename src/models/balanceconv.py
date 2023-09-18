import torch
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, LayerNorm, summary
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, segregate_self_loops, to_networkx
from typing import Any, Dict, List
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor

import networkx as nx
import matplotlib.pyplot as plt


class BalanceConv(MessagePassing):
    def __init__(self) -> None:
        super().__init__(aggr="add")

    def _calculate_flows(self, Din, l, Pin, Pout, E=0.95, K=0.03, rho=0.68):
        """Укрупненно рассчитывает потоки газа по трубам.

        Args:
            Din (Tensor): внутренний диаметр труб газопровода, мм
            Pin, Pout (Tensor): абсолютные давления газа, МПа
            K: эквивалентная шероховатость труб, мм
            rho: плотность газа при с.у., кг/м3
            l (Tensor): протяженность газопровода, км
        """
        lambda_res = 0.067 * torch.pow(2 * K / Din, 0.2)

        # Коэф-т гидр. сопротивления
        lambdah = lambda_res / pow(E, 2)
        delta = torch.tensor(rho / 1.20445)
        Tav = torch.tensor(280.0)  # Условно средняя температура, К
        Zav = torch.tensor(0.92)  # Условно коэф-т сжимаемости, безр.

        flows = (
            3.32
            * 1e-6
            * torch.pow(Din, 2.5)
            * torch.sign(torch.subtract(Pin, Pout))
            * torch.sqrt(
                (torch.abs(torch.pow(Pin, 2) - torch.pow(Pout, 2)))
                / (lambdah * delta * Tav * Zav * l)
                + 1e-6
            )
        )

        return flows

    def forward(
        self, P: Tensor, edge_index: Adj, edge_attr: Tensor, node_attr: Tensor
    ) -> Tensor:
        self.flow = "source_to_target"
        flows_in = self.propagate(
            edge_index=edge_index, p=P, edge_attr=edge_attr, node_attr=node_attr
        )

        self.flow = "target_to_source"
        flows_out = self.propagate(
            edge_index=edge_index, p=P, edge_attr=edge_attr, node_attr=node_attr
        )

        nodes_imbalance = torch.subtract(flows_in, flows_out)
        return flows_in, nodes_imbalance

    def message(
        self, p_j: Tensor, p_i: Tensor, node_attr: OptTensor, edge_attr: OptTensor
    ) -> Tensor:
        """Вычисляет небалансы газа в узлах графа.

        Args:
            p_j (Tensor): Значения давления газа в source nodes, МПа.
            p_i (Tensor): Значения давления газа в target nodes, МПа.
            edge_attr (OptTensor): Характеристики дуг:
              - протяженность, км;
              - внутренний диаметр трубы, мм.

        Returns:
            Tensor: ...
        """
        Pin = p_j if self.flow == "source_to_target" else p_i
        Pout = p_i if self.flow == "source_to_target" else p_j

        flows = self._calculate_flows(
            Din=edge_attr[..., 1],
            l=edge_attr[..., 0],
            Pin=Pin.view(-1),
            Pout=Pout.view(-1),
        ).view(-1, 1)

        return flows
