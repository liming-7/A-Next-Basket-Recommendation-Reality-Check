import numpy as np
import torch


def mae(predict, truth):
    return torch.mean(torch.abs(predict - truth)).item()


def mse(predict, truth):
    return torch.mean((predict - truth) ** 2).item()


def rmse(predict, truth):
    return torch.sqrt(torch.mean((predict - truth) ** 2)).item()


def phr(predict, truth):
    pass

def baskethit(predict, truth):
    pass


def recall(predict, truth, top_k=5):
    """
    Args:
        predict (Tensor): shape (batch_size, items_total)
        truth (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    _, predict_indices = predict.topk(k=top_k)
    predict, truth = predict.new_zeros(predict.shape).scatter_(1, predict_indices, 1).long(), truth.long()
    tp, t = ((predict == truth) & (truth == 1)).sum(-1), truth.sum(-1)
    return (tp.float() / (t.float() + 1e-7)).mean().item()


def precision(predict, truth, top_k=5):
    """
    Args:
        predict (Tensor): shape (batch_size, items_total)
        truth (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    _, predict_indices = predict.topk(k=top_k)
    predict, truth = predict.new_zeros(predict.shape).scatter_(1, predict_indices, 1).long(), truth.long()
    tp, p = ((predict == truth) & (truth == 1)).sum(-1), predict.sum(-1)
    return (tp.float() / (p.float() + 1e-7)).mean().item()


def f1_score(predict: torch.Tensor, truth: torch.Tensor, top_k=5):
    """
    Args:
        predict (Tensor): shape (batch_size, items_total)
        truth (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    _, predict_indices = predict.topk(k=top_k)
    predict, truth = predict.new_zeros(predict.shape).scatter_(1, predict_indices, 1).long(), truth.long() #fill 1 in predicted  indices
    tp, t, p = ((predict == truth) & (truth == 1)).sum(-1), truth.sum(-1), predict.sum(-1)
    precision, recall = tp.float() / (p.float() + 1e-7), tp.float() / (t.float() + 1e-7)
    return (2 * precision * recall / (precision + recall + 1e-7)).mean().item()


def dcg(predict, truth, top_k):
    """
    Args:
        predict: (batch_size, items_total)
        truth: (batch_size, items_total)
        top_k:

    Returns:

    """
    _, predict_indices = predict.topk(k=top_k)
    gain = truth.gather(-1, predict_indices)  # (batch_size, top_k)
    return (gain.float() / torch.log2(torch.arange(top_k, device=predict.device).float() + 2)).sum(-1)  # (batch_size,)


def ndcg(predict, truth, top_k):
    """
    Args:
        predict: (batch_size, items_total)
        truth: (batch_size, items_total)
        top_k:

    Returns:

    """
    dcg_score = dcg(predict, truth, top_k)
    idcg_score = dcg(truth, truth, top_k)
    return (dcg_score / idcg_score).mean().item()


if __name__ == '__main__':
    predict = torch.tensor([
        [0.9, 0.6, 0.1, 0.5, 0.4],
        [0.2, 0.3, 0.6, 0.4, 0.8],
    ])

    truth = torch.tensor([
        [1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0]
    ])

    print(recall(predict, truth, top_k=3))  # 2 / 3, 1 / 2
    print(precision(predict, truth, top_k=3))  # 2 / 3, 1 / 3
    print(f1_score(predict, truth, top_k=3))  # 2 / 3, 2 / 5
    print(ndcg(predict, truth, top_k=3))