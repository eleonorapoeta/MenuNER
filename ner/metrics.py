def calculate_metrics(y_true, y_preds):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for k in range(0, len(y_true)):
        for i, _ in enumerate(y_preds[k]):
            if y_preds[k][i] == 'MENU':
                if y_true[k][i] == 'MENU':
                    tp += 1
                else:
                    fp += 1
            else:
                if y_true[k][i] == 'O':
                    tn += 1
                else:
                    fn += 1
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision == 0 and recall == 0:
        f1_score = 0
    else:

        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score
