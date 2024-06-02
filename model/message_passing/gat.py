from typing import Optional
from argparse import Namespace

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import leaky_relu
from torch_geometric.utils import softmax
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import GATConv

from dropout.base import BaseDropout
from model.message_passing.pretreatment import ModelPretreatment


class GATLayer(GATConv):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_strategy: BaseDropout,
        activation: Module,
        args: Namespace,
    ):

        super(GATLayer, self).__init__(
            in_channels,
            out_channels,
            args.attention_heads,
            concat=False,
        )

        self.pt = ModelPretreatment(args.add_self_loops, args.normalize)
        self.activation = activation
        self.drop_strategy = drop_strategy
        
    def forward(self, x: Tensor, edge_index: Adj):

        x = self.drop_strategy.apply_feature_mat(x, self.training)
        x = self.lin(x)
        x = self.activation(x)

        x_src = x_dst = x.view(-1, self.heads, self.out_channels)
        x = (x_src, x_dst)

        alpha_src = (x_src*self.att_src).sum(dim=-1)
        alpha_dst = (x_dst*self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        edge_index, _ = self.drop_strategy.apply_adj_mat(edge_index, None, self.training)
        edge_index, _ = self.pt.pretreatment(x_src.size(0), edge_index, x_src.dtype)

        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: Tensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        
        alpha = alpha_j + alpha_i
        alpha = leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        
        x_j = alpha.unsqueeze(-1) * x_j
        x_j = self.drop_strategy.apply_message_mat(x_j, self.training)

        return x_j