import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, Parameter
from torch_geometric.nn import GINConv,GCNConv,GATConv, GINEConv, TransformerConv, global_add_pool
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConvWithM(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.lin(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)

        out += self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCNMBlock(torch.nn.Module):
    def __init__(self, indim, hiddendim):
        super(GCNMBlock, self).__init__()
        self.conv = GCNConvWithM(indim, hiddendim)
        self.bn = torch.nn.BatchNorm1d(hiddendim)

    def forward(self, x, edge_index,edge_attr):
        x = F.relu(self.conv(x, edge_index))
        x = self.bn(x)
        return x


class GINBlock(torch.nn.Module):
    def __init__(self, indim, hiddendim):
        super(GINBlock, self).__init__()
        self.conv = GINConv(
            Sequential(Linear(indim, hiddendim), ReLU(), Linear(hiddendim, hiddendim)))
        self.bn = torch.nn.BatchNorm1d(hiddendim)

    def forward(self, x, edge_index,edge_attr):
        x = F.relu(self.conv(x, edge_index))
        x = self.bn(x)
        return x

class GCNBlock(torch.nn.Module):
    def __init__(self, indim, hiddendim):
        super(GCNBlock, self).__init__()
        self.conv = GCNConv(indim, hiddendim)
        self.bn = torch.nn.BatchNorm1d(hiddendim)

    def forward(self, x, edge_index,edge_attr):
        x = F.relu(self.conv(x, edge_index))
        x = self.bn(x)
        return x

class GATBlock(torch.nn.Module):
    def __init__(self, indim, hiddendim):
        super(GATBlock, self).__init__()
        self.conv = GATConv(indim, hiddendim)
        self.bn = torch.nn.BatchNorm1d(hiddendim)

    def forward(self, x, edge_index,edge_attr):
        x = F.relu(self.conv(x, edge_index))
        x = self.bn(x)
        return x


class GINEBlock(torch.nn.Module):
    def __init__(self, indim, hiddendim):
        super(GINEBlock, self).__init__()
        self.conv = GINEConv(
            Sequential(Linear(indim, hiddendim), ReLU(), Linear(hiddendim, hiddendim)), edge_dim=10)
        self.bn = torch.nn.BatchNorm1d(hiddendim)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = self.bn(x)
        return x


class TransConvBlock(torch.nn.Module):
    def __init__(self, indim, hiddendim):
        super(TransConvBlock, self).__init__()
        self.conv = TransformerConv(indim, hiddendim, edge_dim=10)
        self.bn = torch.nn.BatchNorm1d(hiddendim)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = self.bn(x)
        return x


class Graph_Encoder(nn.Module):
    def __init__(self, encoder_type, layer_num, in_channels, basic_channels, dropout):
        super(Graph_Encoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.encoder_type = encoder_type
        self.in_channels = in_channels
        self.basic_channels = basic_channels

        self.layer_num = layer_num
        self.fc = Linear(basic_channels * 3, basic_channels*8)
        self.encoder = self.makelayers()

    def makelayers(self):
        layers = []
        if self.encoder_type == 'GIN':
            layers.append(GINBlock(self.in_channels, self.basic_channels * (self.layer_num + 2)))
            for i in range(self.layer_num - 1):
                layers.append(
                    GINBlock(self.basic_channels * (self.layer_num - i + 2),
                             self.basic_channels * (self.layer_num - i+1)))
        elif self.encoder_type == 'GCN':
            layers.append(GCNBlock(self.in_channels, self.basic_channels * (self.layer_num + 2)))
            for i in range(self.layer_num - 1):
                layers.append(
                    GCNBlock(self.basic_channels * (self.layer_num - i + 2),
                              self.basic_channels * (self.layer_num - i+1)))
        elif self.encoder_type == 'GAT':
            layers.append(GATBlock(self.in_channels, self.basic_channels * (self.layer_num + 2)))
            for i in range(self.layer_num - 1):
                layers.append(
                    GATBlock(self.basic_channels * (self.layer_num - i + 2),
                              self.basic_channels * (self.layer_num - i+1)))
        elif self.encoder_type == 'GINE':
            layers.append(GINEBlock(self.in_channels, self.basic_channels * (self.layer_num + 2)))
            for i in range(self.layer_num - 1):
                layers.append(
                    GINEBlock(self.basic_channels * (self.layer_num - i + 2),
                              self.basic_channels * (self.layer_num - i+1)))
        elif self.encoder_type == 'TransConv':
            layers.append(TransConvBlock(self.in_channels, self.basic_channels * (self.layer_num + 2)))
            for i in range(self.layer_num - 1):
                layers.append(
                    TransConvBlock(self.basic_channels * (self.layer_num - i + 2),
                                   self.basic_channels * (self.layer_num - i+1)))
        elif self.encoder_type == 'GCNwithMessagepassing':
            layers.append(GCNMBlock(self.in_channels, self.basic_channels * (self.layer_num + 2)))
            for i in range(self.layer_num - 1):
                layers.append(
                    GCNMBlock(self.basic_channels * (self.layer_num - i + 2),
                              self.basic_channels * (self.layer_num - i+1)))

        return nn.ModuleList(layers)
    
    '''def dgl_split(self, bg, feats):
        max_num_nodes = int(bg.batch_num_nodes().max())
        batch = torch.cat([torch.full((1, x.type(torch.int)), y) for x, y in zip(bg.batch_num_nodes(), range(bg.batch_size))], dim=1).reshape(-1).type(torch.long).to(bg.device)
        cum_nodes = torch.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])
        idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats
        out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])
        return out'''
    
    def dgl_split(self, bg, feats):
        max_num_nodes = int(bg.batch_num_nodes().max())
        batch = torch.cat([torch.full((1, x.type(torch.int)), y) for x, y in zip(bg.batch_num_nodes(), range(bg.batch_size))],dim=1).reshape(-1).type(torch.long).to(bg.device)
        cum_nodes = torch.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])
        idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats
        out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])
        # 新增：生成 mask
        mask = torch.zeros((bg.batch_size, max_num_nodes), dtype=torch.bool, device=bg.device)
        for i, num_nodes in enumerate(bg.batch_num_nodes()):
            mask[i, :num_nodes] = False  # 有效节点置False，后面的padding节点是True

        return out, mask


    def forward(self,bg_d):
        # x = bg_d.ndata['atom']  # 获取每个节点的特征，形状为 (num_nodes, 44)
        x = torch.cat([bg_d.ndata['atom'], bg_d.ndata['lap_pos_enc']], dim=-1)  # 66+8
        src, dst = bg_d.edges()
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = bg_d.edata['bond']  # 获取每条边的特征，形状为 (num_edges, 10)
        # batch = torch.cat([torch.full((1, x.type(torch.int)), y) for x, y in zip(bg_d.batch_num_nodes(), range(bg_d.batch_size))], dim=1).reshape(-1).type(torch.long).to(bg_d.device)
        for layer in self.encoder:
            x = layer(x, edge_index, edge_attr)
        # x = global_add_pool(x, batch)  # 对全图的点嵌入进行池化并返回图嵌入
        x,mask = self.dgl_split(bg_d, x) #torch.Size([64, 46, 96])
        x = self.dropout(F.relu(self.fc(x))) #torch.Size([64, 43, 256])
        # x = self.dropout(x)
        # x = x.view(x.size(0), self.basic_channels, -1)
        return x,mask
