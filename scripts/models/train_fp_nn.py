import time
import pickle as pkl
import pandas as pd
import argparse
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.optim import Adam




def load_data(df_path):
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
    mlb = MultiLabelBinarizer(classes=all_labels)
    y = mlb.fit_transform(data['summarizations'])
    print(f"y.shape: {y.shape}")

    # create X. This is done this way to prevent strange issue with pd dataframe & sparse matrix
    X = np.zeros((n_samples, n_features))
    for i, fp in enumerate(data['fingerprint']):
        X[i, :] = np.array(list(fp))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"X.shape: {X.shape}")

    # save cid
    cid = data["cid"].values

    return X, y, cid, mlb



class MultilabelClassifier(nn.Module):
    def __init__(self, d_input=2048, d_hidden_1=0, d_hidden_2=0, d_hidden_3=0, d_output=1544):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        if (d_hidden_1 == 0) and (d_hidden_2 == 0) and (d_hidden_3 == 0):
            self.fc1 = nn.Linear(d_input, d_output)
            self.n_hidden_layers = 0
        elif (d_hidden_1 != 0) and (d_hidden_2 == 0) and (d_hidden_3 == 0):
            self.fc1 = nn.Linear(d_input, d_hidden_1)
            self.fc2 = nn.Linear(d_hidden_1, d_output)
            self.n_hidden_layers = 1
        elif (d_hidden_1 != 0) and (d_hidden_2 != 0) and (d_hidden_3 == 0):
            self.fc1 = nn.Linear(d_input, d_hidden_1)
            self.fc2 = nn.Linear(d_hidden_1, d_hidden_2)
            self.fc3 = nn.Linear(d_hidden_2, d_output)
            self.n_hidden_layers = 2
        elif (d_hidden_1 != 0) and (d_hidden_2 != 0) and (d_hidden_3 != 0):
            self.fc1 = nn.Linear(d_input, d_hidden_1)
            self.fc2 = nn.Linear(d_hidden_1, d_hidden_2)
            self.fc3 = nn.Linear(d_hidden_2, d_hidden_3)
            self.fc4 = nn.Linear(d_hidden_3, d_output)
            self.n_hidden_layers = 3
        else:
            raise ValueError("Invalid number of hidden layers.")

    def forward(self, x):
        x = self.dropout(x)
        if self.n_hidden_layers == 0:
            x = self.fc1(x)
        elif self.n_hidden_layers == 1:
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)
        elif self.n_hidden_layers == 2:
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.fc3(x)
        else:
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.fc3(x)
            x = self.dropout(x)
            x = self.fc4(x)
        return x
    


def load_model_device(d_input=2048, d_hidden_1=None, d_hidden_2=None, d_hidden_3=None, d_output=1544):
    model = MultilabelClassifier(d_input=d_input, d_hidden_1=d_hidden_1, d_hidden_2=d_hidden_2, d_hidden_3=d_hidden_3, d_output=d_output)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU.")
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    model.to(device)

    return model, device


def multilabel_loss(loss_func, pred, target):
    losses = 0
    for i in range(pred.shape[0]):
        losses += loss_func(pred[i], target[i])
    return losses


def create_dataloader(X, y, cid=None, batch_size=32):
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    if cid is not None:
        cid_tensor = torch.from_numpy(cid).int()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, cid_tensor)
    else:
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_model(model, X, y, train, valid, fold, results_df, lr_rate=0.001, epochs=10, batch_size=32, save_path = "", device="cpu"):
    print("Training model...")
    optimizer = Adam(model.parameters(), lr=lr_rate)

    loss_func = nn.BCEWithLogitsLoss()
    losses = []

    for epoch in range(epochs):
        avg_epoch_train_loss = 0
        for i, (batch_X, batch_y) in enumerate(create_dataloader(X[train], y[train], batch_size=batch_size)):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # forward pass
            outputs = model(batch_X)
            loss = multilabel_loss(loss_func, outputs, batch_y)
            losses.append(loss.item())
            avg_epoch_train_loss += loss.item()

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f"Train: Fold {fold}, Epoch {epoch}/{epochs}, Batch {i}/{len(train)//batch_size}, Loss {loss.item():.4f}")
                # checkpoint_losses_train.append(loss.item())
        avg_epoch_train_loss /= len(train)//batch_size

        # eval on test with batchloader
        avg_epoch_valid_loss = 0
        for i, (batch_X, batch_y) in enumerate(create_dataloader(X[valid], y[valid], batch_size=batch_size)):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            with torch.no_grad():
                outputs = model(batch_X)
                # forward pass
                loss = multilabel_loss(loss_func, outputs, batch_y)
                
            avg_epoch_valid_loss += loss.item()
        avg_epoch_valid_loss /= len(valid)//batch_size
        print(f"\nValidation: Fold {fold}, Epoch {epoch}/{epochs}, Mean Loss {avg_epoch_valid_loss:.4f}")

        # save results to df
        results_df = pd.concat([results_df, pd.DataFrame({"fold": fold, "epoch": epoch, "train_loss": avg_epoch_train_loss, "valid_loss": avg_epoch_valid_loss}, index=[0])], ignore_index=True)
        
        # save model each epoch
        torch.save(model.state_dict(), f"{save_path}/fold_{fold}_epoch_{epoch}.pth")

        # save losses
        np.save(f"{save_path}/fold_{fold}_epoch_{epoch}_train_losses.npy", losses)

    return results_df




def test_model(model, X, y, cid, mlb, epoch, kfold, batch_size=32, device="cpu", save_path=""):
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

def roc_pr_auc_scores(preds_df, targets_df, best_kfold, best_epoch):
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
    agg_metrics_df.to_csv(f"test_metrics_agg_k{best_kfold}_e{best_epoch}.csv", index=False)

    # I know there's a better way to do this, but this works
    indiv_metrics_df = pd.DataFrame(columns=preds_df.iloc[:, 1:].columns)
    indiv_metrics_df.loc["roc_auc"] = roc_auc
    indiv_metrics_df.loc["avg_prec"] = avg_prec
    # pivot
    indiv_metrics_df = indiv_metrics_df.T
    indiv_metrics_df.to_csv(f"test_metrics_indiv_{best_kfold}_e{best_epoch}.csv", index=True)


def main(args):
    t0 = time.time()

    d_input = args.d_input
    d_hidden_1 = args.d_hidden_1
    d_hidden_2 = args.d_hidden_2
    d_hidden_3 = args.d_hidden_3
    d_output = args.d_output
    kfolds = args.k_folds
    epochs = args.n_epochs
    batch_size = args.batch_size

    # save args with unique filename to txt file
    save_path = Path(f"models/fp_nn/di{d_input}_dh1{d_hidden_1}_dh2{d_hidden_2}_dh3{d_hidden_3}_do{d_output}_kcv{kfolds}_e{epochs}_bs{batch_size}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("Saving to", save_path)
    with open(save_path / "args.txt", "w") as f:
        f.write(str(args))
    
    df_path = '../../results/schembl_summs_v5_final_fp.pkl'
    X, y, cid, mlb = load_data(df_path)
    with open(save_path / "mlb.pkl", "wb") as f:
        pkl.dump(mlb, f)

    # hold out test set
    X_train, X_test, y_train, y_test, cid_train, cid_test = train_test_split(X, y, cid, test_size=0.1, random_state=42)
    
    # train model
    results_df = pd.DataFrame(columns=["fold", "epoch", "train_loss", "valid_loss"])
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    for fold, (train, valid) in enumerate(kf.split(X_train, y_train)):
        model, device = load_model_device(d_input=d_input,
                                            d_hidden_1=d_hidden_1,
                                            d_hidden_2=d_hidden_2,
                                            d_hidden_3=d_hidden_3,
                                            d_output=d_output)
        results_df = train_model(model = model,
                                   X = X_train,
                                   y = y_train,
                                   train = train,
                                   valid = valid,
                                   fold = fold, 
                                   results_df = results_df, 
                                   save_path=save_path, 
                                   device=device, 
                                   epochs=epochs, 
                                   batch_size=batch_size)

    results_df.to_csv(save_path / "train_results.csv", index=False)


    # find the epoch and fold with the lowest validation loss
    min_loss = results_df['valid_loss'].min()
    best_idx = int(results_df['valid_loss'].idxmin())
    best_epoch = int(results_df.iloc[best_idx]["epoch"])
    best_fold = int(results_df.iloc[best_idx]["fold"])

    print(f"Best epoch: {best_epoch}", f"Best fold: {best_fold}", f"Best loss: {min_loss}", sep="\n")

    # load best model
    model, device = load_model_device()
    model.load_state_dict(torch.load(f"{save_path}/fold_{best_fold}_epoch_{best_epoch}.pth"))

    # test model to get loss, and save results to csv
    preds_df, targets_df = test_model(model, X_test, y_test, cid_test, mlb, epoch=best_epoch, kfold=best_fold, save_path=save_path, device=device)

    # calculate roc_auc and average_precision scores
    roc_pr_auc_scores(preds_df, targets_df, best_kfold=best_fold, best_epoch=best_epoch)



    print("Done! Thank you for your patience.")
    print(f"Total time: {time.time()-t0:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_input", type=int, default=2048, help="Dimension of input layer.")
    parser.add_argument("--d_hidden_1", type=int, default=0, help="Dimension of first hidden layer.")
    parser.add_argument("--d_hidden_2", type=int, default=0, help="Dimension of second hidden layer.")
    parser.add_argument("--d_hidden_3", type=int, default=0, help="Dimension of third hidden layer.")
    parser.add_argument("--d_output", type=int, default=1544, help="Dimension of output layer.")
    parser.add_argument("--k_folds", type=int, default=10, help="Number of folds for k-fold cross validation.")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    args = parser.parse_args()
    main(args)