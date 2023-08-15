import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_auc_score, average_precision_score


def main():
    preds_df = pd.read_csv("test_results.csv")
    targets_df = pd.read_csv("test_targets.csv")
    print(preds_df.shape)
    print(targets_df.shape)

    # sort by cid
    preds_df = preds_df.sort_values(by=["cid"])
    targets_df = targets_df.sort_values(by=["cid"])
    # drop columns from preds and targets if targets have all 0s
    preds_df = preds_df.loc[:, (targets_df != 0).any(axis=0)]
    targets_df = targets_df.loc[:, (targets_df != 0).any(axis=0)]
    print("After dropping all 0 columns:")
    print(preds_df.shape)
    print(targets_df.shape)

    
    preds = preds_df.iloc[:, 1:].to_numpy()
    targets = targets_df.iloc[:, 1:].to_numpy()    

    # roc_auc_score
    roc_auc = roc_auc_score(targets, preds, average=None)
    macro_roc_auc = roc_auc_score(targets, preds, average="macro")
    weighted_roc_auc = roc_auc_score(targets, preds, average="weighted")   

    # average_precision_score
    avg_prec = average_precision_score(targets, preds, average=None)
    macro_avg_prec = average_precision_score(targets, preds, average="macro")
    weighted_avg_prec = average_precision_score(targets, preds, average="weighted")

    print("Test set results:")
    print(f"Macro ROC AUC: {macro_roc_auc}")
    print(f"Weighted ROC AUC: {weighted_roc_auc}")
    print(f"Macro Average Precision: {macro_avg_prec}")
    print(f"Weighted Average Precision: {weighted_avg_prec}")

    # write results to csv
    agg_metrics_df = pd.DataFrame({"macro_roc_auc": [macro_roc_auc], "weighted_roc_auc": [weighted_roc_auc], "macro_avg_prec": [macro_avg_prec], "weighted_avg_prec": [weighted_avg_prec]})
    agg_metrics_df.to_csv("test_metrics_agg.csv", index=False)

    metrics_df = pd.DataFrame(columns=preds_df.iloc[:, 1:].columns)
    metrics_df.loc["roc_auc"] = roc_auc
    metrics_df.loc["avg_prec"] = avg_prec
    metrics_df.to_csv("test_metrics_indiv.csv", index=False)

if __name__ == '__main__':
    main()