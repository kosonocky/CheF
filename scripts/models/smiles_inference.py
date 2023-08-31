import time
import argparse
import pickle as pkl
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
from torch.optim import Adam
from rdkit import Chem

from train_fp_nn import load_model_device


def get_X_from_smiles(smiles, scaler_path):
    print("Creating dataset...")
    fingerprint = Chem.RDKFingerprint(Chem.MolFromSmiles(smiles))
    n_samples = 1
    n_features = len(fingerprint)
    print(f"n_samples: {n_samples}, n_features: {n_features}")

    # create X. This is done this way to prevent strange issue with pd dataframe & sparse matrix
    X = np.zeros((n_samples, n_features))
    X[0, :] = np.array(list(fingerprint))
    print(f"X.shape: {X.shape}")
    
    # load standard scaler
    with open(scaler_path, "rb") as f:
        scaler = pkl.load(f)

    # scale X
    X = scaler.transform(X)

    return X

def inference(model, X, mlb, cid="", batch_size=1, device="cpu"):
    """
    Inference on single scaled sfingerprint

    """

    print("Inferencing model...")
    model.eval()
    preds = []
    X_tensor = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        outputs = model(X_tensor)

        # save preds and targets
        preds.append(outputs.cpu().numpy())

    preds = np.vstack(preds)

    # sigmoid activate preds
    preds = 1 / (1 + np.exp(-preds))

    # create df with cid, pred or target in rows
    # and in each column, the label is a column with the value being the probability
    preds_df = pd.DataFrame(columns = ["labels", "preds"])
    preds_df["labels"] = mlb.classes_
    preds_df["preds"] = preds[0]
    # preds_df.reset_index(inplace=True)

    return preds_df


def main(args):
    smiles = args.smiles
    if args.cid != "":
        cid = args.cid
    else:
        print("cid not provided, using smiles as cid")
        cid = smiles
    
    t0 = time.time()
    save_path = Path(f"inference/{cid}")
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = "models/fp_nn/di-2048_dh1-512_dh2-256_dh3-0_do-1543_kcv-5_e-10_bs-32"
    
    X = get_X_from_smiles(smiles, scaler_path=f"{best_model_path}/scaler.pkl")
    
    with open(f"{best_model_path}/mlb.pkl", "rb") as f:
        mlb = pkl.load(f)
    
    # best_model_path = "models/fp_nn/di2048_dh11796_dh20_dh30_do1544_kcv5_e10_bs32/best_model.pth"
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

    # test model to get loss, and save results to csv
    preds_df = inference(model, X, mlb, cid, device=device)

    # transpose to get labels as rows
    # preds_df = preds_df.T

    # save results to csv, sorted by probability
    preds_df = preds_df.sort_values(by="preds", ascending=False)
    preds_df.to_csv(save_path / "predictions.csv", index=False)
    
    print("Done! Thank you for your patience.")
    print(f"Total time: {time.time()-t0:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", help="smiles string of molecule")
    parser.add_argument("--cid", default="", help="cid of molecule")
    args = parser.parse_args()
    main(args)