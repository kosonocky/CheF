import pickle as pkl
from pathlib import Path

import argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, auc, PrecisionRecallDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


def main(args):
    # labels_ = ["antibacterial", "antibiotic", "bactericide",]
    # label = "antibacterial_all"
    # labels_ = ["antiviral", "viral", "anti-viral",]
    # label = "antiviral_all"
    # labels_ = ["electroluminescent", "light-emitting", "emitting"]
    # label = "electroluminescent_all"
    labels_ = ["hepatitis", "hcv", "hbv"]
    label = "hepatitis_all"
    


    model_name = "lr"
    print(f"\nTraining {model_name} for {label}\n")

    df = pd.read_pickle("schembl_summs_v4_final_with_fingerprint_and_chemberta.pkl").reset_index(drop=True)
    # create column label if any of the strings in labels_ are in the set of summarizations
    df[label] = df['summarizations'].apply(lambda x: 1 if any([label_ in x for label_ in labels_]) else 0)

    fp_x = np.array(df["fingerprint"].tolist())
    fp_y = np.array(df[label].tolist())
    cb_x = np.array(df["features"].tolist())
    cb_y = np.array(df[label].tolist())
    

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for x, y, name in zip([fp_x, cb_x], [fp_y, cb_y], ["fp", "cb"]):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42, n_jobs=40, C=0.001))
        ])

        print('Model: ', model)
        print('Data: ', name)
        # create save dir
        save_path = Path(label, name, model_name)
        save_path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(5, 5))
        fig_prc, ax_prc = plt.subplots(figsize=(5, 5))
        tprs = []
        aucs = []
        precisions = []
        aps = []
        mean_fpr = np.linspace(0, 1, 100) # mean_fpr
        
        for fold, (train, test) in enumerate(cv.split(x, y)):
            print(f"Fold {fold}")

            # train the model with 5-fold cross validation
            model.fit(x[train], y[train])

            # save train and test set predictions & probs for each fold
            train_pred = model.predict(x[train])
            train_prob = model.predict_proba(x[train])
            test_pred = model.predict(x[test])
            test_prob = model.predict_proba(x[test])

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
                model,
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


            viz_2 = PrecisionRecallDisplay.from_estimator(
                model,
                x[test],
                y[test],
                name=f'PR fold {fold}',
                alpha=0.3,
                lw=1,
                ax=ax_prc,
                plot_chance_level=(fold == 4),
            )
            
            interp_precision = np.interp(mean_fpr, viz_2.recall[::-1], viz_2.precision[::-1])
            interp_precision[0] = 1.0
            precisions.append(interp_precision)

            aps.append(viz_2.average_precision)

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
            xlim=[0, 1],
            ylim=[0, 1],
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
        ax.set_title(f"ROC Curves {name}+{model_name}\n(Positive label = '{label}')\nGround Truth P: {sum(y)} N: {len(y) - sum(y)}\n")
    

        # save figs. Make sure not to crop legend text on right
        fig.savefig(save_path / 'roc.png', dpi=300, bbox_inches='tight')

    
        # # plot mean precision recall curve
        mean_precision = np.mean(precisions, axis=0)
        mean_precision[-1] = 0.0
        mean_prc_auc = auc(mean_fpr, mean_precision)
        std_prc_auc = np.std(aps)

        ax_prc.plot(
            mean_fpr,
            mean_precision,
            color='b',
            label=r'Mean PR (AUC = %0.2f $\pm$ %0.2f)' % (mean_prc_auc, std_prc_auc),
            lw=2,
            alpha=.8
        )

        ax_prc.set(
            xlim=[0, 1],
            ylim=[0, 1],
            xlabel='Recall',
            ylabel='Precision',
            # title=f"Mean ROC curve with variability\n(Positive label '{label}')"
        )

        # plot variability
        std_precision = np.std(precisions, axis=0)
        precisions_upper = np.minimum(mean_precision + std_precision, 1)
        precisions_lower = np.maximum(mean_precision - std_precision, 0)
        ax_prc.fill_between(
            mean_fpr,
            precisions_lower,
            precisions_upper,
            color='grey',
            alpha=.2,
            label=r'$\pm$ 1 std. dev.'
        )

        # Remove variability from legend
        handles, labels = ax_prc.get_legend_handles_labels()
        handles = handles[:-3] + handles[-2:-1] + handles[-3:-2]
        labels = labels[:-3] + labels[-2:-1] + labels[-3:-2]
        

        # make legend outside plot to right. Make sure text isn't cropped
        ax_prc.legend(
            handles,
            labels,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
        )

        # add title with number of true positives and negatives
        ax_prc.set_title(f"PR Curves {name}+{model_name}\n(Positive label = '{label}')\nGround Truth P: {sum(y)} N: {len(y) - sum(y)}\n")

        # save figs. Make sure not to crop legend text on right
        fig_prc.savefig(save_path / 'pr.png', dpi=300, bbox_inches='tight')

        # save metrics
        with open(save_path / 'metrics.txt', 'w') as f:
            f.write(f"Ground Truth P: {sum(y)}\n")
            f.write(f"Ground Truth N: {len(y) - sum(y)}\n")
            f.write(f"Mean ROC AUC: {mean_auc}\n")
            f.write(f"Std ROC AUC: {std_auc}\n")
            f.write(f"Mean PR AUC: {mean_prc_auc}\n")
            f.write(f"Std PR AUC: {std_prc_auc}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, default='opioid')
    
    args = parser.parse_args()
    main(args)