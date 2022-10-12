import torch
from torch import Tensor
import torch.nn as nn
from interaction_network import InteractionNetwork as IN

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, C):
        return self.layers(C)

class TCN(nn.Module):
    def __init__(self, node_indim, edge_indim):
        super(TCN, self).__init__()
        self.in_w1 = IN(node_indim, edge_indim,
                        node_outdim=node_indim, edge_outdim=edge_indim,
                        hidden_size=80)
        self.in_w2 = IN(node_indim, edge_indim,
                        node_outdim=node_indim, edge_outdim=edge_indim,
                        hidden_size=80)

        self.W = MLP(edge_indim*3, 1, 80)

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Tensor) -> Tensor:

        # re-embed the graph twice with add aggregation
        x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)

        x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)

        # combine all edge features, use to predict edge weights
        initial_edge_attr = torch.cat([edge_attr, edge_attr_1,
                                       edge_attr_2], dim=1)

        edge_weights = torch.sigmoid(self.W(initial_edge_attr))
        return edge_weights

