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




def load_data(df_path):
    print("Creating dataset...")
    df = pd.read_pickle(df_path)
    data = df[['fingerprint', 'summarizations', "cid"]]
    data["summarizations"] = data["summarizations"].map(set)

    # create set of all labels
    all_labels = Counter()
    for labels in data['summarizations']:
        all_labels.update(labels)
    # list of most common keys
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
    def __init__(self, ):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1544)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    


def load_model_device():
    model = MultilabelClassifier()
    
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


def create_dataloader(X, y, batch_size=32):
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train(model, X, y, lr_rate=0.001, epochs=10, batch_size=32, kfolds=10, save_path = "", device="cpu"):
    print("Training model...")
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    optimizer = Adam(model.parameters(), lr=lr_rate)
    # n_total_steps = X.shape[0]

    loss_func = nn.BCEWithLogitsLoss()

    results_df = pd.DataFrame(columns=["fold", "epoch", "train_loss", "valid_loss"])

    for fold, (train, valid) in enumerate(kf.split(X, y)):
        losses = []
        # checkpoint_losses_train = []

        for epoch in range(epochs):
            avg_epoch_train_loss = 0
            # print("\nTraining fold", fold, "epoch", epoch+1)
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
                    print(f"Train: Fold {fold}, Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train)//batch_size}, Loss {loss.item():.4f}")
                    # checkpoint_losses_train.append(loss.item())
            avg_epoch_train_loss /= len(train)//batch_size

            # eval on test with batchloader
            avg_epoch_valid_loss = 0
            # checkpoint_losses_valid = []
            for i, (batch_X, batch_y) in enumerate(create_dataloader(X[valid], y[valid], batch_size=batch_size)):
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                with torch.no_grad():
                    outputs = model(batch_X)
                    # forward pass
                    loss = multilabel_loss(loss_func, outputs, batch_y)
                    # checkpoint_losses_valid.append(loss.item())
                    #         
                avg_epoch_valid_loss += loss.item()
            avg_epoch_valid_loss /= len(valid)//batch_size
            print(f"\nValidation: Fold {fold}, Epoch {epoch+1}/{epochs}, Mean Loss {loss.item():.4f}\n")

            # save results to df
            results_df = results_df.append({"fold": fold, "epoch": epoch+1, "train_loss": avg_epoch_train_loss, "valid_loss": avg_epoch_valid_loss}, ignore_index=True)
            
            # save model each epoch
            torch.save(model.state_dict(), f"{save_path}/fold_{fold}_epoch_{epoch+1}.pth")

    return losses, results_df




def test(model, X, y, cid, mlb, epoch, kfold, batch_size=32, device="cpu", save_path=""):
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

    # convert preds and targets to labels
    preds_labels = mlb.inverse_transform(preds)
    targets_labels = mlb.inverse_transform(targets)

    # save results to csv
    results = pd.DataFrame({"cid": cids, "preds": preds, "targets": targets, "preds_labels": preds_labels, "targets_labels": targets_labels})
    results.to_csv(save_path / "test_results.csv", index=False)

    
    


def main():
    df_path = '../../results/schembl_summs_v5_final_fp.pkl'
    save_path = Path("models/fp_nn")
    save_path.mkdir(parents=True, exist_ok=True)
    
    X, y, cid, mlb = load_data(df_path)

    # save mlb
    with open(save_path / "mlb.pkl", "wb") as f:
        pkl.dump(mlb, f)

    # hold out test set
    X_train, X_test, y_train, y_test, cid_train, cid_test = train_test_split(X, y, cid, test_size=0.1, random_state=42)
    
    model, device = load_model_device()
    losses, results_df = train(model, X_train, X_test, save_path=save_path, device=device, epochs=3, batch_size=32, kfolds=10)

    # save losses
    np.save(save_path / "losses.npy", losses)
    results_df.to_csv(save_path / "results.csv", index=False)

    # find the epoch and fold with the lowest validation loss
    best_epoch = results_df.groupby("fold")["valid_loss"].idxmin().values[0]
    best_fold = results_df.iloc[best_epoch]["fold"]

    # load best model
    model.load_state_dict(torch.load(f"{save_path}/fold_{best_fold}_epoch_{best_epoch+1}.pth"))
    test(model, X_test, y_test, cid_test, mlb, epoch=best_epoch, kfold=best_fold, save_path=save_path)

    print("Done! Thank you for your patience.")


if __name__ == '__main__':
    main()