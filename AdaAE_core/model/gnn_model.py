import torch
from AdaAE_core.search_space.utils import conv_map, act_map
import torch.nn as nn


class NodeFeatureEmbeddingLayerTwibot(nn.Module):
    def __init__(self, hidden_dim, numerical_feature_size=5, categorical_feature_size=3, des_feature_size=768,
                 tweet_feature_size=768, dropout=0.3):
        super(NodeFeatureEmbeddingLayerTwibot, self).__init__()
        self.numerical_feature_size = numerical_feature_size
        self.categorical_feature_size = categorical_feature_size
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.numerical_feature_linear = nn.Sequential(
            nn.Linear(numerical_feature_size, hidden_dim // 4),
            self.activation
        )

        self.categorical_feature_linear = nn.Sequential(
            nn.Linear(categorical_feature_size, hidden_dim // 4),
            self.activation
        )

        self.des_feature_linear = nn.Sequential(
            nn.Linear(des_feature_size, hidden_dim // 4),
            self.activation
        )

        self.tweet_feature_linear = nn.Sequential(
            nn.Linear(tweet_feature_size, hidden_dim // 4),
            self.activation
        )

        self.total_feature_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )
        self.init_weights()
    def init_weights(self):
        # the following code could init the module created by nn.Sequential()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        category_prop = x[:, :3]
        num_prop = x[:, 3:8]
        des_tensor = x[:, 8:776]
        tweet_tensor = x[:, 776:]
        num_prop = self.numerical_feature_linear(num_prop)
        category_prop = self.categorical_feature_linear(category_prop)
        des_tensor = self.des_feature_linear(des_tensor)
        tweet_tensor = self.tweet_feature_linear(tweet_tensor)
        x = torch.cat((num_prop, category_prop, des_tensor, tweet_tensor), dim=1)
        x = self.total_feature_linear(x)
        return x


class NodeFeatureEmbeddingLayerMGTAB(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super(NodeFeatureEmbeddingLayerMGTAB, self).__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.numerical_feature_linear = nn.Sequential(
            nn.Linear(10, hidden_dim // 8),
            self.activation
        )
        self.categorical_feature_linear = nn.Sequential(
            nn.Linear(10, hidden_dim // 8),
            self.activation
        )

        self.des_feature_linear = nn.Sequential(
            nn.Linear(768, 3 * hidden_dim // 4),
            self.activation
        )
        self.total_feature_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )
        self.init_weights()
    def init_weights(self):
        # the following code could init the module created by nn.Sequential()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        d = self.des_feature_linear(x[:, -768:].to(torch.float32))
        n = self.numerical_feature_linear(x[:, [4,6,7,8,10,11,12,13,14,15]].to(torch.float32))
        c = self.categorical_feature_linear(x[:, [1,2,3,5,9,16,17,18,19,20]].to(torch.float32))
        h = torch.cat([d, n, c], dim=1)
        h = self.total_feature_linear(h)
        return h


class GnnModel(torch.nn.Module):
    def __init__(self,
                 sample_architecture,
                 args):

        super(GnnModel, self).__init__()
        self.sample_architecture = sample_architecture
        self.args = args
        self.linear_input_dim = 128
        if 'twibot' in self.args.dataset.lower():
            self.node_feature_embedding_layer = NodeFeatureEmbeddingLayerTwibot(self.linear_input_dim)
            self.num_edge_type = 2
        else:
            self.node_feature_embedding_layer = NodeFeatureEmbeddingLayerMGTAB(self.linear_input_dim)
            self.num_edge_type = 7
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers_types = []
        self.activation_operators = []
        self.dropout = nn.Dropout(p=self.args.gnn_drop_out)
        heterotype = ['rgt', 'hgt', 'simplehgn', 'rgcn']

        for layer in range(self.args.gnn_layers):
            # 层间维度匹配技巧
            if layer == 0:
                input_dim = self.linear_input_dim
            else:
                input_dim = hidden_dimension
            convolution_type = self.sample_architecture[layer * 2 + 0]
            attention_type, aggregator_type, hidden_dimension = convolution_type.split('-')
            if attention_type in heterotype:
                self.conv_layers_types.append('hetero')
            else:
                self.conv_layers_types.append('homo')
            hidden_dimension = int(hidden_dimension)
            conv = conv_map(attention_type, aggregator_type, input_dim, hidden_dimension, self.args.bias,
                            self.args.temperature, self.num_edge_type)
            self.conv_layers.append(conv)

            activation_type = self.sample_architecture[layer * 2 + 1]
            act = act_map(activation_type)
            self.activation_operators.append(act)

        self.residual_linear = nn.Linear(self.linear_input_dim, hidden_dimension)
        self.projection_head = nn.Sequential(nn.Linear(hidden_dimension, hidden_dimension),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dimension, 2))

    def forward(self, data):
        x = data.x
        x = self.node_feature_embedding_layer(x)
        x = self.dropout(x)
        residual = x
        for layer in range(self.args.gnn_layers):
            x = self.conv_layers[layer](x, data.edge_index, data.edge_type)
            x = self.activation_operators[layer](x)
        residual = self.residual_linear(residual)
        x = x + residual
        x = self.projection_head(x)
        return x

    def forward_gumbel(self, data, gumbel_softmax_sample_ret_list, sample_candidate_index_list):
        x = data.x
        x = self.node_feature_embedding_layer(x)
        residual = x
        for layer in range(self.args.gnn_layers):
            x = self.conv_layers[layer](x, data.edge_index, data.edge_type) * \
                gumbel_softmax_sample_ret_list[layer * 2 + 0][0][sample_candidate_index_list[layer * 2 + 0]]
            x = self.activation_operators[layer](x) * gumbel_softmax_sample_ret_list[layer * 2 + 1][0][
                sample_candidate_index_list[layer * 2 + 1]]
        residual = self.residual_linear(residual)
        x = x + residual
        x = self.projection_head(x)
        return x