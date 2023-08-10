import pickle as pkl
from pathlib import Path

import argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, auc
# from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def main(args):
    label = args.label

    fp_df = pd.read_pickle("../../results/schembl_summs_v4_final_with_fingerprint_arr.pkl")
    cb_df = pd.read_pickle("../../results/schembl_summs_v4_final_with_chemberta.pkl")
    # drop na values\
    fp_df = fp_df.dropna(subset=['fingerprint'])
    cb_df = cb_df.dropna(subset=['features'])

    # create column label if label is in "summarizations" column
    fp_df[label] = fp_df['summarizations'].apply(lambda x: 1 if label in x else 0)
    cb_df[label] = cb_df['summarizations'].apply(lambda x: 1 if label in x else 0)

    fp_x = np.array(fp_df["fingerprint"].tolist())
    fp_y = np.array(fp_df[label].tolist())
    cb_x = np.array(cb_df["features"].tolist())
    cb_y = np.array(cb_df[label].tolist())
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fp_rfc = RandomForestClassifier(random_state=42, n_jobs=40)
    cb_rfc = RandomForestClassifier(random_state=42, n_jobs=40)

    fp_models = [fp_rfc]
    cb_models = [cb_rfc]

    # train each model with 5-fold cross validation and then test on the test set
    # save the model and the test results

    for fp_model in fp_models:
        print('\nModel: ', fp_model)
        # create save dir
        save_path = Path(label, "fp", fp_model.__class__.__name__)
        save_path.mkdir(parents=True, exist_ok=True)

        roc_fig, roc_ax = plt.subplots(figsize=(5, 5))
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100) # mean_fpr

        for fold, (train, test) in enumerate(cv.split(fp_x, fp_y)):
            print(f"Fold {fold}")

            # train the model with 5-fold cross validation
            fp_model.fit(fp_x[train], fp_y[train])

            # save train and test set predictions & probs for each fold
            fp_train_pred = fp_model.predict(fp_x[train])
            fp_train_prob = fp_model.predict_proba(fp_x[train])
            fp_test_pred = fp_model.predict(fp_x[test])
            fp_test_prob = fp_model.predict_proba(fp_x[test])

            # merge with df and save df (without fingerprint) to csv
            fp_train_df = fp_df.iloc[train]
            fp_train_df['pred'] = fp_train_pred
            fp_train_df['prob'] = fp_train_prob[:, 1]
            fp_test_df = fp_df.iloc[test]
            fp_test_df['pred'] = fp_test_pred
            fp_test_df['prob'] = fp_test_prob[:, 1]
            fp_train_df[["cid", label, "pred", "prob"]].to_csv(save_path / f"train_fold_{fold}.csv", index=False)
            fp_test_df[["cid", label, "pred", "prob"]].to_csv(save_path / f"test_fold_{fold}.csv", index=False)
            
            # plot roc curve
            viz = RocCurveDisplay.from_estimator(
                fp_model,
                fp_x[test],
                fp_y[test],
                name=f'ROC fold {fold}',
                alpha=0.3,
                lw=1,
                ax=roc_ax,
                plot_chance_level=(fold == 4),
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)


        # plot mean roc curve
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        roc_ax.plot(
            mean_fpr, 
            mean_tpr, 
            color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2,
            alpha=.8
        )
        
        # plot variability
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        roc_ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color='grey',
            alpha=.2,
            label=r'$\pm$ 1 std. dev.'
        )

        # plot chance line
        roc_ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel='False Positive Rate',
            ylabel='True Positive Rate',
            # title=f"Mean ROC curve with variability\n(Positive label '{label}')"
        )


        # save figs
        roc_fig.savefig(save_path / 'roc.png', dpi=300)

        # save metrics
        with open(save_path / 'metrics.txt', 'w') as f:
            f.write(f"Mean ROC AUC: {mean_auc}\n")
            f.write(f"Std ROC AUC: {std_auc}\n")
        

    # print("\nChemberta")

    # for cb_model in cb_models:
    #     print('\nModel: ', cb_model)

    #     # train the model with 5-fold cross validation
    #     # cv_results = cross_validate(cb_model, X=cb_x_train, y=cb_y_train, cv=5, scoring='roc_auc', return_estimator=True)
    #     cv_results = cross_validate(cb_model, X=cb_x, y=cb_y, cv=5, scoring='roc_auc', return_estimator=True)

    #     # display the results
    #     print('Mean Test ROC AUC: ', cv_results['test_score'].mean())
    #     print('Std Test ROC AUC: ', cv_results['test_score'].std())

    #     # test the model on the entire set
    #     cb_y_pred = cv_results['estimator'][0].predict(cb_x)

    #     # get prob
    #     cb_y_prob = cv_results['estimator'][0].predict_proba(cb_x)[:, 1]

    #     # get metrics for the test set
    #     cb_accuracy = accuracy_score(cb_y, cb_y_pred)
    #     cb_precision = precision_score(cb_y, cb_y_pred)
    #     cb_recall = recall_score(cb_y, cb_y_pred)
    #     cb_f1 = f1_score(cb_y, cb_y_pred)
    #     cb_roc_auc = roc_auc_score(cb_y, cb_y_pred)

    #     # display the metrics
    #     print('Accuracy: ', cb_accuracy)
    #     print('Precision: ', cb_precision)
    #     print('Recall: ', cb_recall)
    #     print('F1: ', cb_f1)
    #     print('ROC AUC: ', cb_roc_auc)


    #     # confusion matrix
    #     conf = confusion_matrix(cb_y, cb_y_pred)

    #     # roc curve
    #     roc_ = roc_curve(cb_y, cb_y_prob)


    #     # plot
    #     fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    #     sns.heatmap(conf, annot=True, fmt='d', ax=ax[0])
    #     ax[0].set_title('Confusion Matrix')
    #     ax[0].set_xlabel('Predicted')
    #     ax[0].set_ylabel('Actual')
    #     ax[0].set_xticklabels(['No', 'Yes'])
    #     ax[0].set_yticklabels(['No', 'Yes'])
    #     ax[1].plot(roc_[0], roc_[1])
    #     ax[1].plot([0, 1], [0, 1], 'k--')
    #     ax[1].set_title('ROC Curve')
    #     ax[1].set_xlabel('False Positive Rate')
    #     ax[1].set_ylabel('True Positive Rate')

    #     # save all data, model, results, and plot to folder
    #     save_path = Path(label, "cb", cb_model.__class__.__name__)
    #     save_path.mkdir(parents=True, exist_ok=True)
    #     with open(save_path / 'model.pkl', 'wb') as f:
    #         pkl.dump(cv_results['estimator'][0], f)
    #     with open(save_path / 'results.txt', 'w') as f:
    #         f.write(f'Mean Test ROC AUC across splits: {cv_results["test_score"].mean()}\n')
    #         f.write(f'Std Test ROC AUC across splits: {cv_results["test_score"].std()}\n\n')
    #         f.write(f"Test metrics using the best model from cross validation\n")
    #         f.write(f'Accuracy: {cb_accuracy}\n')
    #         f.write(f'Precision: {cb_precision}\n')
    #         f.write(f'Recall: {cb_recall}\n')
    #         f.write(f'F1: {cb_f1}\n')
    #         f.write(f'ROC AUC: {cb_roc_auc}\n')

    #     # append predictions
    #     cb_df[f'{label}_pred'] = cb_y_pred
    #     cb_df[f'{label}_prob'] = cb_y_prob
    #     cb_df.to_csv(save_path / 'pred.csv', index=False)
    #     fig.savefig(save_path / 'plot.png')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, default='opioid')
    
    args = parser.parse_args()
    main(args)