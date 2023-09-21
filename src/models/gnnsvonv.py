import torch
from torch.nn import Sequential, Linear, ELU, LeakyReLU
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, LayerNorm
from torch_geometric.data import Data
from typing import Any, Dict, List
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor


class GNNSConv(MessagePassing):
    """Сверточный слой графовой нейронной сети, выполняющий преобразование латентного представления вершин графа."""

    def __init__(
        self,
        latent_dim: int,
        num_edge_features: int,
        alpha: float = 0.01,
        hidden_layers=[8],
        aggr="mean",
    ) -> None:
        """
        Args:
            in_channels: число фичей в вершинах графа
            alpha: параметр для учета величины обновления X
        """
        super().__init__(aggr=aggr)
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers

        layers = []
        l = 2 * latent_dim + num_edge_features
        layers.append(LayerNorm(l))
        for size in hidden_layers:
            layers.append(Linear(l, size))
            layers.append(LeakyReLU())
            l = size
        layers.append(Linear(l, latent_dim))
        self.mlp_in = Sequential(*layers)

        layers = []
        l = 2 * latent_dim + num_edge_features
        layers.append(LayerNorm(l))
        for size in hidden_layers:
            layers.append(Linear(l, size))
            layers.append(LeakyReLU())
            l = size
        layers.append(Linear(l, latent_dim))
        self.mlp_out = Sequential(*layers)

        layers = []
        l = latent_dim * 3 + num_edge_features
        layers.append(LayerNorm(l))
        for size in hidden_layers:
            layers.append(Linear(l, size))
            layers.append(LeakyReLU())
            l = size
        layers.append(Linear(l, latent_dim))
        self.mlp_psi = Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(
        self, X: Tensor, edge_index: Adj, node_attr: Tensor, edge_attr: Tensor
    ) -> Tensor:
        """
        Args:
            X: Латентное представление данных в вершинах графа, [num_nodes, in_channels]
            edge_index: Разреженная матрица смежности, [2, num_nodes]
            node_attr: Значения параметров в вершинах графа, [num_nodes, num_node_features]
            edge_attr: Значения атрибутов дуг графа, [num_edges, num_edge_features]
        """

        assert (
            X.shape[1] == self.latent_dim
        ), f"Размерность латентного вектора X должна совпадать с размерностью `latent_dim`, получено X.shape={X.shape}, latent_dim={self.latent_dim}"

        # Формирование сообщения j -> i
        self.flow = "source_to_target"
        phi_in = self.propagate(edge_index, x=X, edge_attr=edge_attr)

        # Формирование сообщения i -> j
        self.flow = "target_to_source"
        phi_out = self.propagate(edge_index, x=X, edge_attr=edge_attr)

        # Объединение сообщений
        inputs = torch.cat([X, node_attr, phi_in, phi_out], dim=-1)
        psi = self.mlp_psi(inputs)
        X = X + self.alpha * psi
        return X

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        """
        Args:
            x_i: Target node features [num_edges, num_features]
            x_j: Source node features [num_edges, num_features]
            edge_attr: Source node features [num_edges, num_edge_features]
        """
        mlp = self.mlp_in if self.flow == "source_to_target" else self.mlp_out
        tmp = torch.cat([x_j, x_i, edge_attr], dim=-1)
        out_message = mlp(tmp)
        return out_message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(aggr={self.aggr}, latent_dim={self.latent_dim}, alpha={self.alpha}, hidden_layers={self.hidden_layers})"
