import argparse
import random
import torch
import numpy as np
import os
from AdaAE_core.auto_model import AutoModel


def arg_parse():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser(description='AdaAE_core Arguments.')
    parser.add_argument('--data_root', type=str, default="./data/")
    parser.add_argument('--dataset', dest='dataset', default='MGTAB', help='Dataset')  # MGTAB   Twibot-20
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gnn_layers', type=int, default=2)
    parser.add_argument('--gnn_drop_out', type=float, default=0.3)
    parser.add_argument('--bias', type=str2bool, default=True)
    parser.add_argument('--temperature', type=float, default=1e-3,
                        help='the temperature parameter of gumbel softmax trick')
    parser.add_argument('--search_epoch', type=int, default=100, help='the number of search epoch for gumbel')
    parser.add_argument('--learning_rate_gumbel', type=float, default=0.1, help='the learning rate for gumbel')
    parser.add_argument('--l2_regularization_strength_gumbel', type=float, default=0.1,
                        help='the regularization strength for gumbel')
    parser.add_argument('--train_epoch', type=int, default=200,
                        help='the number of train epoch for model sampled by gumbel')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='the learning rate for contrastive learning')
    parser.add_argument('--l2_regularization_strength', type=float, default=5e-3,
                        help='the regularization strength for contrastive learning')
    parser.add_argument('--return_top_k', type=int, default=5, help='the number of top model for testing')
    parser.add_argument('--attention', type=str, default=None, help='fix attention type')
    parser.add_argument('--activation', type=str, default=None, help='fix activation type')
    parser.add_argument('--hidden_dimension', type=str, default=None, help='fix hidden dimension')
    parser.add_argument('--aggregation', type=str, default=None, help='fix aggregation type')
    return parser.parse_args()


def sample_mask(idx, l):
    """Create mask."""
    mask = torch.zeros(l)
    mask[idx] = 1
    return torch.as_tensor(mask, dtype=torch.bool)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mask_train_label(keep_rate, train_mask):
    print("mask train dataset, keep_rate: ", keep_rate)
    train_idx = torch.where(train_mask == 1)[0]
    train_num_nodes = train_idx.shape[0]
    perm = torch.randperm(train_num_nodes)
    keep_nodes = int(keep_rate * train_num_nodes)
    train_idx = train_idx[perm][:keep_nodes]
    mask = torch.zeros_like(train_mask, dtype=torch.bool)
    mask[train_idx] = 1
    train_mask = mask.bool()
    return train_mask


def mask_edge(keep_rate, edge_index, edge_type):
    print("mask edge, keep_rate: ", keep_rate)
    edge_num = edge_index.shape[1]
    perm = torch.randperm(edge_num)
    keep_edges = int(keep_rate * edge_num)
    edge_index = edge_index[:, perm][:, :keep_edges]
    edge_type = edge_type[perm][:keep_edges]
    return edge_index, edge_type


def mask_feature(keep_rate, x):
    # drop_prob: probability of feature dropout
    drop_prob = 1 - keep_rate
    drop_mask = torch.empty(
        (x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(x.device)
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def main(seed):
    args = arg_parse()
    args.seed = seed
    if args.dataset == 'Twibot-20':
        graph = torch.load(r'../Data/Twibot-20-2-hops-subgraph-with-text.pt')
        graph.dataset = 'Twibot-20'
        # 这里做参数敏感性实验
        args.learning_rate_gumbel = 0.5
        args.temperature = 0.4
    elif args.dataset == 'MGTAB':
        # MGTAB已经用numpy随机数种子0划分了数据集，无需再次划分
        graph = torch.load(r'../Data/MGTAB.pt')
        graph.dataset = 'MGTAB'
        # 这里做参数敏感性实验
        args.learning_rate_gumbel = 0.1
        args.temperature = 0.1
    else:
        raise ValueError('Dataset not found.')

    # 这里做消融实验
    # args.attention = 'rgt'
    # args.hidden_dimension = '128'
    # args.aggregation = 'mean'
    # args.activation = 'relu'

    # 这里用来做数据有效性实验
    # graph.edge_index, graph.edge_type = mask_edge(0.1, graph.edge_index, graph.edge_type)
    # graph.x = mask_feature(0.1, graph.x)
    # graph.train_mask = mask_train_label(0.1, graph.train_mask)

    print(args)
    set_seed(seed)
    auto_model = AutoModel(graph, args)
    auto_model.search_model()
    metrics = auto_model.derive_target_model()
    return metrics


if __name__ == '__main__':
    test_metrics = []
    seeds = [100]
    # seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for seed in seeds:
        metric = main(seed)
        test_metrics.append(metric)
    # 小数点保留两位
    acc_mean = round(np.mean([x['acc'] for x in test_metrics]), 2)
    acc_std = round(np.std([x['acc'] for x in test_metrics]), 2)
    f1_mean = round(np.mean([x['f1'] for x in test_metrics]), 2)
    f1_std = round(np.std([x['f1'] for x in test_metrics]), 2)
    precision_mean = round(np.mean([x['precision'] for x in test_metrics]), 2)
    precision_std = round(np.std([x['precision'] for x in test_metrics]), 2)
    recall_mean = round(np.mean([x['recall'] for x in test_metrics]), 2)
    recall_std = round(np.std([x['recall'] for x in test_metrics]), 2)
    print(f'acc: {acc_mean}$\pm${acc_std}')
    print(f'f1_score: {f1_mean}$\pm${f1_std}')
    print(f'precision: {precision_mean}$\pm${precision_std}')
    print(f'recall: {recall_mean}$\pm${recall_std}')
    print('---------------------------------------------')