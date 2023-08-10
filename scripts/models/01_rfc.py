import pickle as pkl
from pathlib import Path

import argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


def main(args):
    label = args.label
    print(f"Training RFC for {label}\n")

    df = pd.read_pickle("schembl_summs_v4_final_with_fingerprint_and_chemberta.pkl").reset_index(drop=True)
    print("df len", len(df))
    # create column label if label is in "summarizations" column
    df[label] = df['summarizations'].apply(lambda x: 1 if label in x else 0)
    # print number of labels
    print("labels", df[label].value_counts())

    fp_x = np.array(df["fingerprint"].tolist())
    fp_y = np.array(df[label].tolist())
    cb_x = np.array(df["features"].tolist())
    cb_y = np.array(df[label].tolist())
    

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for x, y, name in zip([fp_x, cb_x], [fp_y, cb_y], ["fp", "cb"]):
        # make standard scaler rfc pipeline
        rfc = Pipeline([
            ('scaler', StandardScaler()),
            ('rfc', RandomForestClassifier(random_state=42, n_jobs=40))
        ])
        

        print('\nModel: ', rfc)
        print('Data: ', name)
        # create save dir
        save_path = Path(label, name, "rfc")
        save_path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(5, 5))
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100) # mean_fpr

        for fold, (train, test) in enumerate(cv.split(x, y)):
            print(f"Fold {fold}")

            # train the model with 5-fold cross validation
            rfc.fit(x[train], y[train])

            # save train and test set predictions & probs for each fold
            train_pred = rfc.predict(x[train])
            train_prob = rfc.predict_proba(x[train])
            test_pred = rfc.predict(x[test])
            test_prob = rfc.predict_proba(x[test])

            # merge with df and save df (without fingerprint) to csv
            train_df = df.iloc[train]
            train_df['pred'] = train_pred
            train_df['prob'] = train_prob[:, 1]
            test_df = df.iloc[test]
            test_df['pred'] = test_pred
            test_df['prob'] = test_prob[:, 1]
            train_df[["cid", label, "pred", "prob"]].to_csv(save_path / f"train_fold_{fold}.csv", index=False)
            test_df[["cid", label, "pred", "prob"]].to_csv(save_path / f"test_fold_{fold}.csv", index=False)
            
            # plot roc curve
            viz = RocCurveDisplay.from_estimator(
                rfc,
                x[test],
                y[test],
                name=f'ROC fold {fold}',
                alpha=0.3,
                lw=1,
                ax=ax,
                plot_chance_level=(fold == 4),
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

            # write file with TP, TN, FP, FN
            tn, fp, fn, tp = confusion_matrix(y[test], test_pred).ravel()
            with open(save_path / f"confusion_matrix_fold_{fold}.txt", "w") as f:
                f.write(f"TP: {tp}\n")
                f.write(f"TN: {tn}\n")
                f.write(f"FP: {fp}\n")
                f.write(f"FN: {fn}\n")
                # plot precision, recall acc, f
                f.write(f"Precision: {precision_score(y[test], test_pred)}\n")
                f.write(f"Recall: {recall_score(y[test], test_pred)}\n")
                f.write(f"Accuracy: {accuracy_score(y[test], test_pred)}\n")
                f.write(f"F1: {f1_score(y[test], test_pred)}\n")
                f.write(f"AUC: {roc_auc_score(y[test], test_pred)}\n")
                
            # plot confusion matrix
            plt_cm, ax_cm = plt.subplots(figsize=(5, 5))
            disp = ConfusionMatrixDisplay.from_predictions(
                y[test],
                test_pred,
                ax=ax_cm,
            )
            disp.ax_.set_title(f"Confusion matrix fold {fold}")
            disp.figure_.savefig(save_path / f"confusion_matrix_fold_{fold}.png")


        # plot mean roc curve
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr, 
            mean_tpr, 
            color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2,
            alpha=.8
        )
        
        # plot chance line
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel='False Positive Rate',
            ylabel='True Positive Rate',
            # title=f"Mean ROC curve with variability\n(Positive label '{label}')"
        )

        # plot variability
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color='grey',
            alpha=.2,
            label=r'$\pm$ 1 std. dev.'
        )

        # show legend in order of folds, then mean, then chance
        handles, labels = ax.get_legend_handles_labels()

        # handles = handles[:-1]
        # labels = labels[:-1]

        handles = handles[:-3] + handles[-2:-1] + handles[-3:-2]
        labels = labels[:-3] + labels[-2:-1] + labels[-3:-2]
        # make legend outside plot to right. Make sure text isn't cropped
        ax.legend(
            handles,
            labels,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
        )

        # add title with number of true positives and negatives
        # ax.set_title(f"Mean ROC curve with variability\n(Positive label '{label}')")
        ax.set_title(f"Mean ROC curve with variability\n(Positive label '{label}')\nTP: {sum(y)} TN: {len(y) - sum(y)}\n")
    

        # save figs. Make sure not to crop legend text on right
        fig.savefig(save_path / 'roc.png', dpi=300, bbox_inches='tight')

        # save metrics
        with open(save_path / 'metrics.txt', 'w') as f:
            f.write(f"Mean ROC AUC: {mean_auc}\n")
            f.write(f"Std ROC AUC: {std_auc}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, default='opioid')
    
    args = parser.parse_args()
    main(args)