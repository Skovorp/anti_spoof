import datetime
from calculate_eer import compute_eer
import numpy as np
from torch.nn import functional as F

def pretty_now():
    return str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3))).strftime("%Y-%m-%d_%H:%M:%S"))


def eer_metric(logits, targets):
    preds_spoof = F.softmax(logits, -1)[:, 0].cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    res = compute_eer(preds_spoof[targets == 0], preds_spoof[targets == 1])[0]
    return res