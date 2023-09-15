import time
import pickle as pkl
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn

from train_fp_nn import load_model_device, create_dataloader, multilabel_loss


def load_data_test(df_path, mlb_path, scaler_path):
    """
    Loads mlb and scaler (this is not used for training)
    """
    print("Creating dataset...")
    df = pd.read_pickle(df_path)
    data = df[['fingerprint', 'summarizations', "cid"]]
    data["summarizations"] = data["summarizations"].map(set)

    # create set of all labels
    all_labels = Counter()
    for labels in data['summarizations']:
        all_labels.update(labels)
    # list of most common keys1
    all_labels = [label for label, _ in all_labels.most_common()]

    n_samples = data.shape[0]
    n_features = round(data['fingerprint'].apply(len).mean())
    n_classes = len(all_labels)
    n_labels = round(data['summarizations'].apply(len).mean())
    print(f"n_samples: {n_samples}, n_features: {n_features}, n_classes: {n_classes}, n_labels: {n_labels}")

    # create multi-label binary matrix
    with open(mlb_path, "rb") as f:
        mlb = pkl.load(f)
    y = mlb.transform(data['summarizations'])
    print(f"y.shape: {y.shape}")

    # create X. This is done this way to prevent strange issue with pd dataframe & sparse matrix
    X = np.zeros((n_samples, n_features))
    for i, fp in enumerate(data['fingerprint']):
        X[i, :] = np.array(list(fp))

    # load standard scaler
    with open(scaler_path, "rb") as f:
        scaler = pkl.load(f)
    X = scaler.transform(X)
    print(f"X.shape: {X.shape}")
    
    # save cid
    cid = data["cid"].values

    return X, y, cid, mlb



def roc_pr_auc_scores(preds_df, targets_df, save_path = ""):
    # sort by cid
    preds_df = preds_df.sort_values(by=["cid"])
    targets_df = targets_df.sort_values(by=["cid"])
    # drop columns from preds and targets if targets have all 0s
    preds_df = preds_df.loc[:, (targets_df != 0).any(axis=0)]
    targets_df = targets_df.loc[:, (targets_df != 0).any(axis=0)]
    
    # assumes CID is first column
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
    agg_metrics_df.to_csv(f"{save_path}/test_metrics_agg.csv", index=False)

    # I know there's a better way to do this, but this works
    indiv_metrics_df = pd.DataFrame(columns=preds_df.iloc[:, 1:].columns)
    indiv_metrics_df.loc["roc_auc"] = roc_auc
    indiv_metrics_df.loc["avg_prec"] = avg_prec
    # pivot
    indiv_metrics_df = indiv_metrics_df.T
    indiv_metrics_df.to_csv(f"{save_path}/test_metrics_indiv.csv", index=True)



def test_model(model, X, y, cid, mlb, batch_size=32, device="cpu", save_path=""):
    """
    Test set eval that saves results to csv (with cid, pred, target, and converted labels)

    """

    print("Testing model...")
    loss_func = nn.BCEWithLogitsLoss()
    model.eval()
    losses = []
    preds = []
    targets = []
    cids = []
    for i, (batch_X, batch_y, batch_cid) in enumerate(create_dataloader(X, y, cid, batch_size=batch_size)):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        with torch.no_grad():
            outputs = model(batch_X)
            # forward pass
            loss = multilabel_loss(loss_func, outputs, batch_y)
            losses.append(loss.item())

            # save preds and targets
            preds.append(outputs.cpu().numpy())
            targets.append(batch_y.cpu().numpy())

            # save cid
            cids.append(batch_cid)

    preds = np.vstack(preds)
    targets = np.vstack(targets)
    cids = np.hstack(cids)

    # sigmoid activate preds
    preds = 1 / (1 + np.exp(-preds))

    # create df with cid, pred or target in rows
    # and in each column, the label is a column with the value being the probability
    preds_df = pd.DataFrame(preds, index=cids, columns=mlb.classes_)
    preds_df.index.name = "cid"
    preds_df.reset_index(inplace=True)

    # add in targets as rows, with probabilities in each column
    targets_df = pd.DataFrame(targets, index=cids, columns=mlb.classes_)
    targets_df.index.name = "cid"
    targets_df.reset_index(inplace=True)

    # print average loss
    print(f"Test loss: {np.mean(losses):.4f}")
    
    # save results to csv
    preds_df.to_csv(save_path / "test_preds.csv", index=False)
    targets_df.to_csv(save_path / "test_targets.csv", index=False)

    return preds_df, targets_df


def main():
    t0 = time.time()
    df_path = '../../results/schembl_summs_v5_final_fp.pkl'
    save_path = Path("models/fp_nn/best_model_test")
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = "models/fp_nn/di-2048_dh1-512_dh2-256_dh3-0_do-1543_kcv-5_e-10_bs-32"
    
    X, y, cid, mlb = load_data_test(df_path, mlb_path=f"{best_model_path}/mlb.pkl", scaler_path=f"{best_model_path}/scaler.pkl")

    # hold out test set
    X_train, X_test, y_train, y_test, cid_train, cid_test = train_test_split(X, y, cid, test_size=0.1, random_state=42)
    
    d_input = int(best_model_path.split("/")[-1].split("_")[0].split("di-")[-1])
    d_hidden_1 = int(best_model_path.split("/")[-1].split("_")[1].split("dh1-")[-1])
    d_hidden_2 = int(best_model_path.split("/")[-1].split("_")[2].split("dh2-")[-1])
    d_hidden_3 = int(best_model_path.split("/")[-1].split("_")[3].split("dh3-")[-1])
    d_output = int(best_model_path.split("/")[-1].split("_")[4].split("do-")[-1])


    # load best model
    model, device = load_model_device(
        d_input = d_input,
        d_hidden_1 = d_hidden_1,
        d_hidden_2 = d_hidden_2,
        d_hidden_3 = d_hidden_3,
        d_output = d_output,
    )
    model.load_state_dict(torch.load(f"{best_model_path}/best_model.pth"))

    # write model architecture to file
    with open(save_path / "best_model_architecture.txt", "w") as f:
        f.write(f"model_path: {best_model_path}")
        f.write("\n")
        f.write(str(model))

    # test model to get loss, and save results to csv
    preds_df, targets_df = test_model(model, X_test, y_test, cid_test, mlb, save_path=save_path, device=device)

    # calculate roc_auc and average_precision scores
    roc_pr_auc_scores(preds_df, targets_df, save_path=save_path)

    print("Done! Thank you for your patience.")
    print(f"Total time: {time.time()-t0:.2f} seconds")


if __name__ == '__main__':
    main()