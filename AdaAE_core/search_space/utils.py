import torch
from torch.nn import Sequential, Linear, ReLU, PReLU, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, RGCNConv, TransformerConv, HGTConv
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Module, LayerNorm, PReLU, Sequential, Linear
from torch import Tensor


def get_net(name, input_dim, out_dim, heads, aggregator_type, bias, num_edge_type):
    if name == 'RGTLayer':
        return RGTLayer(in_channels=input_dim,
                              out_channels=out_dim,
                              num_edge_type=num_edge_type,
                              trans_heads=heads,
                              semantic_head=heads,
                              bias=bias,
                              aggr=aggregator_type)
    elif name == 'HGTLayer':
        return HGTLayer(in_channel=input_dim,
                              out_channel=out_dim,
                              heads=4,
                              bias=bias,
                              aggr=aggregator_type)
    elif name == 'SimpleHGN':
        return SimpleHGN(num_edge_type=num_edge_type,
                               in_channels=input_dim,
                               out_channels=out_dim,
                               rel_dim=100,
                               bias=bias,
                               aggr=aggregator_type)
    elif name == 'GCN':
        return GCNConv(input_dim,
                             out_dim,
                             bias=bias,
                             aggr=aggregator_type,
                             normalize=True)
    elif name == 'SAGE':
        return SAGEConv(input_dim,
                              out_dim,
                              bias=bias,
                              aggr=aggregator_type,
                              normalize=True)
    elif name == 'RGCN':
        return RGCNConv(input_dim,
                              out_dim,
                              num_relations=num_edge_type,
                              num_bases=4,
                              bias=bias,
                              aggr=aggregator_type)
    elif name == 'GAT':
        return GATConv(input_dim,
                             out_dim,
                             bias=bias,
                             aggr=aggregator_type,
                             heads=4,
                             concat=False)
    elif name == 'GIN':
        return GINConv(Sequential(Linear(input_dim, out_dim),
                                  BatchNorm1d(out_dim),
                           ReLU(),
                           Linear(out_dim, out_dim)))

    else:
        raise NotImplementedError

class CoGNN(Module):
    def __init__(self, env_net_type, act_net_type, aggregator_type, input_dim, out_dim, bias, temp, num_edge_type):
        super(CoGNN, self).__init__()
        self.temp = temp
        self.input_layer_norm = LayerNorm(input_dim)
        self.output_layer_norm = LayerNorm(out_dim)
        self.env_net_type = env_net_type
        self.act_net_type = act_net_type
        self.env_net = get_net(name=env_net_type, input_dim=input_dim, out_dim=out_dim, heads=4, aggregator_type=aggregator_type, bias=bias, num_edge_type=num_edge_type)
        self.in_act_net = get_net(name=act_net_type, input_dim=input_dim, out_dim=2, heads=1, aggregator_type=aggregator_type, bias=bias, num_edge_type=num_edge_type)
        self.out_act_net = get_net(name=act_net_type, input_dim=input_dim, out_dim=2, heads=1, aggregator_type=aggregator_type, bias=bias, num_edge_type=num_edge_type)
        self.HETERO_TYPE = ['RGTLayer', 'HGTLayer', 'SimpleHGN', 'RGCN']
    def forward(self, x: Tensor, edge_index: Adj, edge_type: OptTensor = None) -> Tensor:
        if self.act_net_type in self.HETERO_TYPE:
            in_logits = self.in_act_net(x=x, edge_index=edge_index, edge_type=edge_type)  # (N, 2)
            out_logits = self.out_act_net(x=x, edge_index=edge_index, edge_type=edge_type)  # (N, 2)
        else:
            in_logits = self.in_act_net(x=x, edge_index=edge_index)  # (N, 2)
            out_logits = self.out_act_net(x=x, edge_index=edge_index)  # (N, 2)
        temp = self.temp
        in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
        out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)
        edge_weight = self.create_edge_weight(edge_index=edge_index,
                                              keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
        edge_index_temp = edge_index.clone().t()
        edge_index_temp = edge_index_temp[edge_weight > 0].t()
        edge_type_temp = edge_type[edge_weight > 0]
        if self.env_net_type in self.HETERO_TYPE:
            x = self.env_net(x=x, edge_index=edge_index_temp, edge_type=edge_type_temp)
        else:
            x = self.env_net(x=x, edge_index=edge_index_temp)
        return x

    def create_edge_weight(self, edge_index: Adj, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob


def act_map(activation_type):
    if activation_type == "elu":
        act = torch.nn.functional.elu
    elif activation_type == "leaky_relu":
        act = torch.nn.functional.leaky_relu
    elif activation_type == "relu":
        act = torch.nn.functional.relu
    elif activation_type == "relu6":
        act = torch.nn.functional.relu6
    elif activation_type == "sigmoid":
        act = torch.sigmoid
    elif activation_type == "softplus":
        act = torch.nn.functional.softplus
    elif activation_type == "tanh":
        act = torch.tanh
    elif activation_type == "linear":
        act = lambda x: x
    else:
        raise Exception("Wrong activate function")
    return act



def conv_map(attention_type, aggregator_type, input_dim, out_dim, bias, temp=0.5, num_edge_type=2):
    if attention_type == 'gcn':
        conv_layer = CoGNN(env_net_type='GCN', act_net_type='RGTLayer', aggregator_type=aggregator_type, input_dim=input_dim, out_dim=out_dim, bias=bias, temp=temp, num_edge_type=num_edge_type)

    elif attention_type == 'gat':
        conv_layer = CoGNN(env_net_type='GAT', act_net_type='RGTLayer', aggregator_type=aggregator_type, input_dim=input_dim, out_dim=out_dim, bias=bias, temp=temp, num_edge_type=num_edge_type)

    elif attention_type == 'graphsage':
        conv_layer = CoGNN(env_net_type='SAGE', act_net_type='RGTLayer', aggregator_type=aggregator_type, input_dim=input_dim, out_dim=out_dim, bias=bias, temp=temp, num_edge_type=num_edge_type)
    elif attention_type == 'gin':
        conv_layer = CoGNN(env_net_type='GIN', act_net_type='RGTLayer', aggregator_type=aggregator_type, input_dim=input_dim, out_dim=out_dim, bias=bias, temp=temp, num_edge_type=num_edge_type)
    elif attention_type == 'rgcn':
        conv_layer = CoGNN(env_net_type='RGCN', act_net_type='RGTLayer', aggregator_type=aggregator_type, input_dim=input_dim, out_dim=out_dim, bias=bias, temp=temp, num_edge_type=num_edge_type)
    elif attention_type == 'rgt':
        conv_layer = CoGNN(env_net_type='RGTLayer', act_net_type='RGTLayer', aggregator_type=aggregator_type, input_dim=input_dim, out_dim=out_dim, bias=bias, temp=temp, num_edge_type=num_edge_type)
    elif attention_type == 'hgt':
        conv_layer = CoGNN(env_net_type='HGTLayer', act_net_type='RGTLayer', aggregator_type=aggregator_type, input_dim=input_dim, out_dim=out_dim, bias=bias, temp=temp, num_edge_type=num_edge_type)
    elif attention_type == 'simplehgn':
        conv_layer = CoGNN(env_net_type='SimpleHGN', act_net_type='RGTLayer', aggregator_type=aggregator_type, input_dim=input_dim, out_dim=out_dim, bias=bias, temp=temp, num_edge_type=num_edge_type)
    else:
        raise Exception("Wrong conv function")
    return conv_layer


def masked_edge_index(edge_index, edge_mask):
    return edge_index[:, edge_mask]


class SemanticAttention(torch.nn.Module):
    def __init__(self, in_channel, num_head):
        super(SemanticAttention, self).__init__()

        self.in_channel = in_channel
        self.num_head = num_head
        self.multi_head_att_layer = torch.nn.Sequential(
            torch.nn.Linear(in_channel, num_head, bias=False),
            torch.nn.PReLU())

    def forward(self, z):
        t = z.mean(0)
        w = self.multi_head_att_layer(t)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        beta = torch.stack(torch.split(beta, split_size_or_sections=1, dim=2), dim=0)
        output = (beta * z).sum(2)
        output = torch.mean(output, dim=0)
        return output


class RGTLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_edge_type=2, trans_heads=1, semantic_head=4, dropout=0.2,
                 bias=True, aggr='add'):
        super(RGTLayer, self).__init__()
        self.activation = torch.nn.PReLU()
        self.transformer_list = torch.nn.ModuleList()
        for i in range(int(num_edge_type)):
            self.transformer_list.append(
                TransformerConv(in_channels=in_channels, out_channels=out_channels // trans_heads, heads=trans_heads,
                                dropout=dropout,
                                concat=True, bias=bias, aggr=aggr))

        self.num_edge_type = num_edge_type
        self.semantic_attention = SemanticAttention(in_channel=out_channels, num_head=semantic_head)

    def forward(self, x, edge_index, edge_type):
        r"""
        feature: input node features
        edge_index: all edge index, shape (2, num_edges)
        edge_type: same as RGCNconv in torch_geometric
        num_rel: number of relations
        beta: return cross relation attention weight
        agg: aggregation type across relation embedding
        """
        # different edge_index for different edge_type
        edge_index_list = []
        for i in range(self.num_edge_type):
            tmp = masked_edge_index(edge_index, edge_type == i)
            edge_index_list.append(tmp)

        u = self.transformer_list[0](x, edge_index_list[0].squeeze(0)).unsqueeze(1)  # .unsqueeze(1)
        for i in range(1, len(edge_index_list)):
            temp = self.transformer_list[i](x, edge_index_list[i].squeeze(0)).unsqueeze(1)
            u = torch.cat((u, temp), dim=1)
        v = self.semantic_attention(u)
        return v


class HGTLayer(torch.nn.Module):
    def __init__(self, in_channel, out_channel, heads, bias=True, aggr='add'):
        super(HGTLayer, self).__init__()
        # HGTConv的aggr只能是add
        self.HGT_layer = HGTConv(in_channels=in_channel, out_channels=out_channel, heads=heads,
                                 metadata=(['user'], [('user', 'follower', 'user'), ('user', 'following', 'user')]),
                                 bias=bias)

    def forward(self, x, edge_index, edge_type):
        if len(edge_type) == 0:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            edge_index_dict = {('user', 'following', 'user'): edge_index}
            x_dict = {"user": x}
            return self.HGT_layer(x_dict, edge_index_dict)['user']
            # return user_feature
        # for Twibot-20, edge_type = 0 means following, edge_type = 1 means follower
        # for Twibot-22, edge_type = 0 means follower, edge_type = 1 means following
        edge_index = edge_index.t()
        follower_edge_index = edge_index[edge_type == 1]
        follower_edge_index = follower_edge_index.t()
        following_edge_index = edge_index[edge_type == 0]
        following_edge_index = following_edge_index.t()
        x_dict = {"user": x}
        if len(follower_edge_index) == 0:
            edge_index_dict = {('user', 'following', 'user'): following_edge_index}
        elif len(following_edge_index) == 0:
            edge_index_dict = {('user', 'follower', 'user'): follower_edge_index}
        else:
            edge_index_dict = {('user', 'follower', 'user'): follower_edge_index,
                               ('user', 'following', 'user'): following_edge_index}
        x_dict = self.HGT_layer(x_dict, edge_index_dict)
        return x_dict['user']


class SimpleHGN(MessagePassing):
    def __init__(self, in_channels, out_channels, num_edge_type, rel_dim, beta=None, bias=True, aggr='add'):
        super(SimpleHGN, self).__init__(aggr=aggr, node_dim=0)
        self.W = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.W_r = torch.nn.Linear(rel_dim, out_channels, bias=bias)
        self.a = torch.nn.Linear(3 * out_channels, 1, bias=bias)
        self.W_res = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.rel_emb = torch.nn.Embedding(num_edge_type, rel_dim)
        self.beta = beta
        self.leaky_relu = torch.nn.LeakyReLU(0.2)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x, edge_index, edge_type, pre_alpha=None):

        node_emb = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, pre_alpha=pre_alpha)
        output = node_emb + self.W_res(x)
        return output

    def message(self, x_i, x_j, edge_type, pre_alpha, index, ptr, size_i):
        out = self.W(x_j)
        rel_emb = self.rel_emb(edge_type)
        alpha = self.leaky_relu(self.a(torch.cat((self.W(x_i), self.W(x_j), self.W_r(rel_emb)), dim=1)))
        alpha = softmax(alpha, index, ptr, size_i)
        if pre_alpha is not None and self.beta is not None:
            alpha = alpha * (1 - self.beta) + pre_alpha * (self.beta)
        else:
            alpha = alpha
        out = out * alpha.view(-1, 1)
        return out

    def update(self, aggr_out):
        return aggr_out
