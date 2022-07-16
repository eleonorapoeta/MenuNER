import numpy as np
import torch
from tqdm import tqdm
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

def to_numpy(x):
    if type(x) != list:
        x = x.detach().cpu().numpy()
    else:
        for i in range(len(x)):
            x[i] = np.array(x[i])
    return x


def to_device(x, device):
    if type(x) != list:
        x = x.to(device)
    else:
        for i in range(len(x)):
            x[i] = x[i].to(device)
    return x


def evaluation(model, loader, eval_device, epoch=" "):
    eval_loss = 0.
    n_batches = len(loader)

    first_pred = True
    with torch.no_grad():
        for step, (x, y) in enumerate(tqdm(loader, total=len(loader), desc=f"Evaluation {epoch}")):
            model.eval()
            x = to_device(x, eval_device)
            y = to_device(y, eval_device)
            mask = torch.sign(torch.abs(x[1])).to(torch.uint8)
            logits, predictions = model(x)
            log_likelihood = model.crf_module(logits, y, mask=mask, reduction='mean')
            loss = log_likelihood * (-1)

            if first_pred:
                preds = to_numpy(predictions)
                labels = to_numpy(y)
                first_pred = False
            else:
                preds = np.append(preds, to_numpy(predictions), axis=0)
                labels = np.append(labels, to_numpy(y), axis=0)

            eval_loss += loss.item()

    eval_loss = eval_loss / n_batches
    list_labels = [[] for _ in range(labels.shape[0])]
    list_preds = [[] for _ in range(labels.shape[0])]

    idx_to_tag = {0: 'O', 1: 'MENU', 2: 'O'}
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] != 0:  # pad
                list_labels[i].append(idx_to_tag[labels[i][j]])
                list_preds[i].append(idx_to_tag[preds[i][j]])

    report = {
        "loss": eval_loss,
        "precision": precision_score(list_labels, list_preds),
        "recall": recall_score(list_labels, list_preds),
        "f1": f1_score(list_labels, list_preds),
        "report": classification_report(list_labels, list_preds, digits=4),
    }
    print(report['report'])
    return report
