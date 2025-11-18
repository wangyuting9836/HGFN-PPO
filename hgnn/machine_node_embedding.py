# _*_ coding: UTF-8 _*_

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv

from hgnn.hetero_gatv2_conv import HeteroGATv2Conv


class MachineNodeEmbeddingLayer(nn.Module):
    def __init__(self, operation_in_dim, machine_in_dim, o_to_m_edge_in_dim=2, machine_out_dim=10, heads=4):
        super().__init__()
        self.m_gat_conv = HeteroGATv2Conv(in_channels=(operation_in_dim, machine_in_dim), out_channels=machine_out_dim, edge_dim=o_to_m_edge_in_dim,
                                          heads=heads, concat=False)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, train_mask_dict):
        o_to_m_mask = train_mask_dict[('operation', 'o_m', 'machine')]

        m_embedding = self.m_gat_conv(
            x_dict['operation'],
            x_dict['machine'],
            edge_index_dict[('operation', 'o_m', 'machine')][:, o_to_m_mask],
            edge_index_dict[('machine', 'self_loop', 'machine')],
            edge_attr_dict[('operation', 'o_m', 'machine')][o_to_m_mask, :]
        )

        return {"machine": m_embedding.sigmoid()}
