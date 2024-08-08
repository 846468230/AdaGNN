import copy
import torch
from AdaAE_core.model.gnn_model import GnnModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def train_one_epoch(model, optimizer, graph_data, CELoss, args):
    model.train()
    graph_data = graph_data.to(args.device)
    y = graph_data.y
    x = model(graph_data)
    train_x = x[graph_data.train_mask]
    train_y = y[graph_data.train_mask]
    cls_loss = CELoss(train_x, train_y)
    optimizer.zero_grad()
    train_loss = cls_loss
    train_loss.backward()
    optimizer.step()
    model.eval()
    x = model(graph_data)
    val_x = x[graph_data.val_mask]
    val_y = y[graph_data.val_mask]
    val_acc = round(
        accuracy_score(val_y.to('cpu').detach().numpy(), torch.argmax(val_x, dim=-1).to('cpu').detach().numpy()), 5)
    return val_acc


@torch.no_grad()
def test(model, graph_data, args):
    model.eval()
    graph_data = graph_data.to(args.device)
    x = model(graph_data)
    x = x[graph_data.test_mask]
    y = graph_data.y[graph_data.test_mask]
    y_true = y.to('cpu').detach().numpy()
    y_pred_label = torch.argmax(x, dim=-1).to('cpu').detach().numpy()
    test_acc = round(accuracy_score(y_true, y_pred_label), 6)
    print('Test Accuracy: {:.4f}'.format(test_acc))
    test_f1 = round(f1_score(y_true, y_pred_label), 6)
    print('Test F1: {:.4f}'.format(test_f1))
    test_precision = round(precision_score(y_true, y_pred_label), 6)
    print('Test Precision: {:.4f}'.format(test_precision))
    test_recall = round(recall_score(y_true, y_pred_label), 6)
    print('Test Recall: {:.4f}'.format(test_recall))
    return {'acc': test_acc * 100, 'f1': test_f1 * 100, 'precision': test_precision * 100, 'recall': test_recall * 100}


def scratch_train(graph_data, model_component, args):
    model = GnnModel(model_component, args).to(args.device)
    optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                 lr=args.learning_rate,
                                 weight_decay=args.l2_regularization_strength)
    CELoss = torch.nn.CrossEntropyLoss()
    best_val_acc = 0
    best_model = None
    non_increase_count = 0
    for epoch in range(args.train_epoch):
        val_acc = train_one_epoch(model,
                                  optimizer,
                                  graph_data,
                                  CELoss,
                                  args)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            non_increase_count = 0
        else:
            non_increase_count += 1
            if non_increase_count >= 20:
                break
    return best_val_acc, best_model