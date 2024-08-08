import torch.nn.functional as F
from AdaAE_core.search_space.search_space_config import SearchSpace
from AdaAE_core.model.ArchitectureOptimizer import ArchitectureGradientOptimizer
from AdaAE_core.model.logger import gnn_architecture_save, gnn_architecture_load, gnn_architecture_merge
from AdaAE_core.model.test import scratch_train, test
import torch.nn as nn

class AutoModel(object):
    def __init__(self, graph_data, args):
        self.graph_data = graph_data
        self.args = args
        self.search_space = SearchSpace(self.args)
        self.architecture_gradient_optimizer = ArchitectureGradientOptimizer(self.search_space, self.args)

    def search_model(self):
        min_loss = 10000000
        architecture_alpha_list = self.architecture_gradient_optimizer.architecture_alpha_list
        for epoch in range(self.args.search_epoch):
            print(32 * "=")
            print("Search Epoch:", epoch + 1)
            gumbel_softmax_sample_output_list = []
            for architecture_alpha in architecture_alpha_list:
                gumbel_softmax_sample_output_list.append(
                    self.hard_gumbel_softmax_sample(F.softmax(architecture_alpha, dim=-1)))
            sample_candidate_index_list, sample_architecture = self.gnn_architecture_decode(
                gumbel_softmax_sample_output_list)
            print("Sampled Architecture: ", sample_architecture)
            # build_optimize_gnn_model会构建一个GnnModel，然后训练这个GnnModel，再把这个GnnModel作为architecture_gradient_optimizer的一个属性
            self.architecture_gradient_optimizer.build_optimize_gnn_model(sample_architecture, self.graph_data)
            # architecture_gradient_optimizer.forward会调用GnnModel.forward_gumbel，然后返回x
            x = self.architecture_gradient_optimizer(self.graph_data, gumbel_softmax_sample_output_list,
                                                     sample_candidate_index_list)
            self.architecture_gradient_optimizer.optimizer.zero_grad()
            nan_indicator = False
            # 原本的AdaAE_core代码使用的是train_mask，但是我感觉应该是val_mask
            x_train = x[self.graph_data.val_mask]
            y_train = self.graph_data.y[self.graph_data.val_mask]
            loss = nn.CrossEntropyLoss()(x_train, y_train)
            if str(loss.item()) == 'nan':
                nan_indicator = True
            if not nan_indicator:
                print("total_loss:", loss.item())
                min_loss = min(min_loss, loss.item())
                loss.backward()
                self.architecture_gradient_optimizer.optimizer.step()
                best_model = self.architecture_gradient_optimizer.best_alpha_gnn_architecture()
                # 这里的Best Model的含义是所有alpha的值最大的那个model
                print("Best Model:", best_model)
        print(32 * "=")
        print("Search Ending")
        # 逻辑是：当sample出的model不在history中，就加入history的末尾；认为越往后sample出的model越好；返回history的最后k个model
        best_alpha_model_list = self.architecture_gradient_optimizer.get_top_architecture(self.args.return_top_k)
        gnn_architecture_save(self.args, best_alpha_model_list)

    def hard_gumbel_softmax_sample(self, sample_probability):
        hard_gumbel_softmax_sample_output = F.gumbel_softmax(logits=sample_probability,
                                                             tau=self.args.temperature,
                                                             hard=True)
        return hard_gumbel_softmax_sample_output

    def gnn_architecture_decode(self, gumbel_softmax_sample_ret_list):
        candidate_list = []
        candidate_index_list = []
        for i, component_one_hot in enumerate(gumbel_softmax_sample_ret_list):
            component_one_hot = component_one_hot.cpu().detach().numpy().tolist()[0]
            candidate_index = component_one_hot.index(max(component_one_hot))
            candidate_index_list.append(candidate_index)
            component = self.search_space.stack_gnn_architecture[i]
            candidate_list.append(self.search_space.space_dict[component][candidate_index])
        return candidate_index_list, candidate_list

    def derive_target_model(self):
        best_alpha_model_list = gnn_architecture_load(self.args, self.args.gnn_layers)
        print(35 * "=" + " the testing start " + 35 * "=")
        best_val_acc_list = []
        best_model_list = []
        # 加载最后k个model，然后训练，找到最大val_acc对应的model
        for best_alpha_model in best_alpha_model_list:
            val_acc, model, = scratch_train(graph_data=self.graph_data,
                                                model_component=best_alpha_model,
                                                args=self.args)
            best_val_acc_list.append(val_acc)
            best_model_list.append(model)
        best_val_acc = max(best_val_acc_list)
        best_val_index = best_val_acc_list.index(best_val_acc)
        best_model = best_model_list[best_val_index]
        # 测试最大val_acc对应的model
        print("Test the best model: ", best_alpha_model_list[best_val_index])
        metric = test(best_model, self.graph_data, self.args)
        return metric
