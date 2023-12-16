import datetime
from calculate_eer import calculate_eer

def pretty_now():
    return str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3))).strftime("%Y-%m-%d_%H:%M"))


def eer_metric(preds, targets):
    preds_spoof = preds[:, 0]
    return calculate_eer(preds_spoof[targets == 0], preds_spoof[targets == 1])[0]