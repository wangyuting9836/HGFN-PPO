# _*_ coding: UTF-8 _*_
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj
from torch_geometric.utils import softmax


class HeteroGATv2Conv(MessagePassing):

    def __init__(
            self,
            in_channels: Tuple[int, int],
            out_channels: int,
            edge_dim: int,
            heads: int = 2,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            bias: bool = True,
            **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_src_a = Linear(in_channels[0], heads * out_channels, bias=bias, weight_initializer='glorot')
        self.lin_src_b = Linear(in_channels[1], heads * out_channels, bias=bias, weight_initializer='glorot')
        self.lin_dst = Linear(in_channels[1], heads * out_channels, bias=bias, weight_initializer='glorot')

        self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')

        self.att = Parameter(torch.empty(1, heads, out_channels))

        self.self_mlp = nn.Sequential(
            nn.Linear(in_channels[1], heads * out_channels),
            nn.ReLU(),
            nn.Linear(heads * out_channels, heads * out_channels)
        )

        self.gate_linear = Linear(2 * out_channels, 1)

        total_out_channels = out_channels * (heads if concat else 1)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src_a.reset_parameters()
        self.lin_src_b.reset_parameters()
        self.lin_dst.reset_parameters()
        self.lin_edge.reset_parameters()
        for layer in self.self_mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        glorot(self.att)
        zeros(self.bias)


    def forward(  # noqa: F811
            self,
            x_a: Tensor,
            x_b: Tensor,
            edge_index_a2b: Tensor,
            edge_index_b2b: Tensor,
            edge_attr_a2b: Tensor
    ) -> Tensor:

        H, C = self.heads, self.out_channels

        src_x_a_feat = self.lin_src_a(x_a).view(-1, H, C)
        src_x_b_feat = self.lin_src_b(x_b).view(-1, H, C)

        dst_x = self.lin_dst(x_b).view(-1, H, C)
        edge_attr = self.lin_edge(edge_attr_a2b).view(-1, H, C)

        out1 = self.propagate(
            edge_index_a2b,
            x=(src_x_a_feat, dst_x),
            edge_attr=edge_attr,
            size=(src_x_a_feat.size(0), dst_x.size(0)),
        )

        if self.concat:
            out1 = out1.view(-1, self.heads * self.out_channels)
        else:
            out1 = out1.mean(dim=1)

        out2 = self.self_mlp(x_b).view(-1, H, C)

        if self.concat:
            out2 = out2.view(-1, self.heads * self.out_channels)
        else:
            out2 = out2.mean(dim=1)

        gate_input = torch.cat([out1, out2], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_input))
        out = gate * out1 + (1 - gate) * out2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(
            self,
            x_j: Tensor,
            x_i: Tensor,
            edge_attr: Tensor,
            index: Tensor,
            size_i: int,
    ) -> Tensor:

        x = x_i + x_j  # [E, H, C]
        if edge_attr is not None:
            x = x + edge_attr
        x = F.leaky_relu(x, self.negative_slope)
        e = (x * self.att).sum(dim=-1)
        alpha = softmax(e, index, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def message_and_aggregate(self, edge_index: Adj) -> Tensor:
        pass

    def edge_update(self) -> Tensor:
        pass

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
