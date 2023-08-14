import time
import pickle as pkl
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.optim import Adam

from train_fp_nn import load_data, load_model_device, train_model, test_model


def main():
    t0 = time.time()
    df_path = '../../results/schembl_summs_v5_final_fp.pkl'
    save_path = Path("models/fp_nn")
    save_path.mkdir(parents=True, exist_ok=True)
    
    X, y, cid, mlb = load_data(df_path)

    # hold out test set
    X_train, X_test, y_train, y_test, cid_train, cid_test = train_test_split(X, y, cid, test_size=0.1, random_state=42)
    
  
    best_epoch = 3
    best_fold = 3

    # load best model
    model, device = load_model_device()
    model.load_state_dict(torch.load(f"{save_path}/fold_{best_fold}_epoch_{best_epoch+1}.pth"))

    # test model
    test_model(model, X_test, y_test, cid_test, mlb, epoch=best_epoch, kfold=best_fold, save_path=save_path, device=device)


    print("Done! Thank you for your patience.")
    print(f"Total time: {time.time()-t0:.2f} seconds")


if __name__ == '__main__':
    main()