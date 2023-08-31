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

def get_X_from_smiles(smiles):
    fingerprint = Chem.RDKFingerprint(Chem.MolFromSmiles(smiles))
    n_samples = 1
    n_features = len(fingerprint)

    # create X. This is done this way to prevent strange issue with pd dataframe & sparse matrix
    X = np.zeros((n_samples, n_features))
    X[0, :] = np.array(list(fingerprint))
    
    return X

def inference(model, X, mlb, batch_size=32, device="cpu"):
    """
    Inference on single scaled sfingerprint

    """

    print("Inferencing model...")
    model.eval()
    preds = []
    X_tensor = torch.from_numpy(X).float()
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    

    for i, (batch_X) in enumerate(dataloader):
        print(f"Batch {i+1}/{len(dataloader)}", end="\r")
        batch_X = batch_X[0].to(device)

        with torch.no_grad():
            outputs = model(batch_X)
            # add outputs to preds
            preds.append(outputs.cpu().numpy())
            
            
    
    # concatenate each batch of preds
    preds = np.concatenate(preds, axis=0)
    
    # sigmoid activate preds
    preds = 1 / (1 + np.exp(-preds))

    # create df with cid, pred or target in rows
    # and in each column, the label is a column with the value being the probability
    preds_df = pd.DataFrame(columns = mlb.classes_, data=preds)

    return preds_df





def main():
    t0 = time.time()
    
    df = pd.read_csv("../../data/opentargets/opentargets_drugs.csv")
    df = df[df["maximumClinicalTrialPhase"] == 4]
    df = df[["id", "canonicalSmiles"]]
    df.columns = ["id", "smiles"]
    df = df.dropna()
    df = df.drop_duplicates(subset="smiles")
    df = df.reset_index(drop=True)

    save_path = Path(f"inference/opentargets_drugs")
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = "models/fp_nn/di-2048_dh1-512_dh2-256_dh3-0_do-1543_kcv-5_e-10_bs-32"
    
    # get unscaled X from smiles
    df["X"] = df["smiles"].apply(lambda x: get_X_from_smiles(x))
    # load standard scaler and scale X
    with open(f"{best_model_path}/scaler.pkl", "rb") as f:
        scaler = pkl.load(f)
    df["X"] = df["X"].apply(lambda x: scaler.transform(x))
    X = np.vstack(df["X"].values)
    df = df.drop(columns=["X"])
    
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
    preds_df = inference(model, X, mlb, device=device)

    print(preds_df.shape)

    df = pd.concat([df, preds_df], axis=1)

    # # save results to csv, sorted by probability
    df.to_csv(save_path / "predictions.csv", index=False)
    df.to_pickle(save_path / "predictions.pkl")
    
    print("Done! Thank you for your patience.")
    print(f"Total time: {time.time()-t0:.2f} seconds")


if __name__ == '__main__':
    main()