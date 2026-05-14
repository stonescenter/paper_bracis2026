import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
from torch.nn import Linear, Sequential, ReLU

from sklearn.metrics import roc_auc_score


#https://github.com/pyg-team/pytorch_geometric/discussions/7524

class GraphSAGEForLinkPrediction(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels): 
        super(GraphSAGEForLinkPrediction, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class LinkPredictorBase(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LinkPredictorBase, self).__init__()
        self.lin1 = torch.nn.Linear(2 * in_channels, hidden_channels) # Concatenates two embeddings
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, edge_index, x):
        # x is the node embeddings from GraphSAGE
        # edge_index contains the indices of node pairs to evaluate
        row, col = edge_index
        # Concatenate the embeddings of source and destination nodes
        z = torch.cat([x[row], x[col]], dim=-1)
        z = self.lin1(z)
        z = F.relu(z)
        z = self.lin2(z)
        return z
    
class Link_Prediction(torch.nn.Module):
    def __init__(self, type, in_channels, hidden_channels, out_channels, heads=2):
      #super(Link_Prediction, self).__init__()
      super().__init__()
      torch.manual_seed(1234567)

      if type == "GCN":
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
      elif type == "SAGE":
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

      elif type == "GAT": 
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=0.2
        )

        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False, 
            dropout=0.2
        )
        
    #def encode(self, data):
    def encode(self, x, edge_index):
      #x, edge_index, _ = data.x, data.edge_index, data.edge_attr

      x = self.conv1(x, edge_index).relu()
      x = F.dropout(x, p=0.5, training=self.training)
      return self.conv2(x, edge_index)

    #def decode(self, z, data):
    def decode(self, z, edge_label_idx):
      #edge_label_idx = data.edge_label_idx
      return (z[edge_label_idx[0]] * z[edge_label_idx[1]]).sum(dim=-1) # innner product
    
    def decode_all(self, z):
      prob_adj = z @ z.t() # matrix multiplication
      return (prob_adj > 0).nonzero(as_tuple=False).t()
    


class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class EdgeAwareLinkPredictor(torch.nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        torch.manual_seed(1234567)
        self.mlp = Sequential(
            Linear(node_dim * 2 + edge_dim, 128),
            ReLU(),
            Linear(128, 1)
        )

    def forward(self, z, edge_label_index, edge_attr):
        src, dst = edge_label_index
        out = torch.cat([z[src], z[dst], edge_attr], dim=-1)
        return self.mlp(out).view(-1)

class LinkPredictorGAT(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.mlp = Sequential(
            Linear(in_channels * 2, 128),
            ReLU(),
            Linear(128, 1)
        )

    def forward(self, z, edge_label_index):
        src, dst = edge_label_index
        out = torch.cat([z[src], z[dst]], dim=-1)
        return self.mlp(out).view(-1)
    
class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=0.2,
            #add_self_loops=True
        )

        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False, 
            dropout=0.2,
            #add_self_loops=True
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x
    

    