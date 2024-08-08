import torch
import torch.nn as nn
from torch.autograd import Variable
from AdaAE_core.model.gnn_model import GnnModel


class ArchitectureGradientOptimizer(torch.nn.Module):
    def __init__(self, search_space, args):
        super(ArchitectureGradientOptimizer, self).__init__()

        self.search_space = search_space
        self.args = args

        # build learnable architecture alpha parameter
        self.architecture_alpha_list = []

        for component in self.search_space.stack_gnn_architecture:
            candidates = self.search_space.space_dict[component]

            architecture_alpha = Variable(torch.Tensor(1, len(candidates))).to(self.args.device)
            architecture_alpha.requires_grad = True

            nn.init.uniform_(architecture_alpha)
            self.architecture_alpha_list.append(architecture_alpha)

        # optimizer for learnable architecture alpha parameter
        self.optimizer = torch.optim.Adam(self.architecture_alpha_list,
                                          lr=self.args.learning_rate_gumbel,
                                          weight_decay=self.args.l2_regularization_strength_gumbel)

        self.best_architecture_history = []


    def build_optimize_gnn_model(self, sample_architecture, graph_data):
        model = GnnModel(sample_architecture, self.args).to(self.args.device)
        cl_optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                        lr=self.args.learning_rate,
                                        weight_decay=self.args.l2_regularization_strength)
        model.train()
        for gnn_epoch in range(self.args.train_epoch):
            graph_data = graph_data.to(self.args.device)
            x = model(graph_data)
            labels = graph_data.y[graph_data.train_mask]
            x = x[graph_data.train_mask]
            ce_loss = model.ce_loss(x, labels)
            cl_optimizer.zero_grad()
            total_loss = ce_loss
            total_loss.backward()
            cl_optimizer.step()
        self.cl_model = model


    def forward(self, data, gumbel_softmax_sample_ret_list, sample_candidate_index_list):
        x = self.cl_model.forward_gumbel(data, gumbel_softmax_sample_ret_list, sample_candidate_index_list)
        return x

    def best_alpha_gnn_architecture(self):
        best_alpha_architecture_temp = []
        for i, component_vec in enumerate(self.architecture_alpha_list):
            component_vec = component_vec.cpu().detach().numpy().tolist()[0]
            best_alpha_index = component_vec.index(max(component_vec))
            component = self.search_space.stack_gnn_architecture[i]
            operator = self.search_space.space_dict[component][best_alpha_index]
            best_alpha_architecture_temp.append(operator)
        best_alpha_architecture = []
        for layer in range(self.args.gnn_layers):
            convolution_type = best_alpha_architecture_temp[layer * 2 + 0]
            attention_type, aggregator_type, hidden_dimension = convolution_type.split('-')
            best_alpha_architecture += [attention_type, aggregator_type, hidden_dimension]

            activation_type = best_alpha_architecture_temp[layer * 2 + 1]
            best_alpha_architecture.append(activation_type)
        if best_alpha_architecture not in self.best_architecture_history:
            self.best_architecture_history.append(best_alpha_architecture)

        return best_alpha_architecture

    def get_top_architecture(self, top_k):
        best_alpha_gnn_architecture_list = self.best_architecture_history[-top_k:]
        return best_alpha_gnn_architecture_list
