import torch
import torch.nn as nn
from torch.nn import Parameter, Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
class GCN(MessagePassing):

    def __init__(self,in_channels, out_channels):
        super().__init__(aggr='add')  # 聚合方式为求和
        self.lin = Linear(in_channels, out_channels, bias =False)
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()
    def reset_parameters(self) -> None:
       self.lin.reset_parameters()
       self.bias.data.zero_

    


    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # 线性变换
        x = self.lin(x)
        # 归一化
        row, col = edge_index
        deg = degree(col, x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # 传递消息
        out = self.propagate(edge_index, x=x,norm = norm)




    def messsage(self,x_j, norm):
        return norm.view(-1, 1) * x_j

