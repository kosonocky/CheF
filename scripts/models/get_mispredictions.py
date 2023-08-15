import pandas as pd
import numpy as np


def main():

    preds_df = pd.read_csv("test_results.csv")
    targets_df = pd.read_csv("test_targets.csv")

    # sort by cid
    preds_df = preds_df.sort_values(by=["cid"])
    targets_df = targets_df.sort_values(by=["cid"])

    # drop columns from preds and targets if targets have all 0s
    preds_df_bin = preds_df.loc[:, (targets_df != 0).any(axis=0)]
    targets_df_bin = targets_df.loc[:, (targets_df != 0).any(axis=0)]

    # convert to 0 or 1 based on cutoff
    cutoff = 0.1
    preds_df_bin.iloc[:, 1:] = preds_df_bin.iloc[:, 1:].applymap(lambda x: 1 if x >= cutoff else 0)
    
    preds_probs = preds_df.iloc[:, 1:].to_numpy()
    preds = preds_df_bin.iloc[:, 1:].to_numpy(dtype=np.int32)
    targets = targets_df_bin.iloc[:, 1:].to_numpy(dtype=np.int32)

    # create entry of mispredicted labels
    # format is (cid, column_name, pred, target)
    mispreds = []
    correct = []
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if preds[i, j] != targets[i, j]:
                mispreds.append((preds_df.iloc[i, 0], preds_df_bin.columns[j+1], preds_probs[i,j], preds[i, j], targets[i, j]))
            
            # also append if both are 1
            if preds[i, j] == 1 and targets[i, j] == 1:
                correct.append((preds_df.iloc[i, 0], preds_df_bin.columns[j+1], preds_probs[i,j], preds[i, j], targets[i, j]))

    # write to csv
    mispreds_df = pd.DataFrame(mispreds, columns=["cid", "column_name", "pred_prob", "pred", "target"])
    mispreds_df.to_csv("mispredictions.csv", index=False)

    correct_df = pd.DataFrame(correct, columns=["cid", "column_name", "pred_prob", "pred", "target"])
    correct_df.to_csv("correct_predictions.csv", index=False)


if __name__ == '__main__':
    main()