from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_log_error
import scipy

def kfold_cv_thres_fps(X, y, model, k=10, n=100):
    fs_col = X.columns
    mse_list = []
    mae_list = []
    rmsle_list = []
    r2_list = []
    adj_r2_list = []
    pcc_list = []
    pval_list = []
    first = True
    sort_fi = None
    feat_list = []
    gt_pred = []
    gt_pred_inv = []
    fi_all = None
    while True:
        kf = KFold(n_splits=k)
        kf.get_n_splits(X)
        y_pred = []
        y_true = []
        y_pred_inv = []
        y_true_inv = []
        fi_ar = np.zeros(fs_col.shape)
        X = X[fs_col]
        feat_list.insert(0, np.array(fs_col))
        for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            X_scaler = StandardScaler()
            y_scaler = StandardScaler()
            X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y[train_idx]
            X_test, y_test = X.iloc[test_idx].reset_index(drop=True), y[test_idx]
            X_train = pd.DataFrame(X_scaler.fit_transform(X_train), columns=fs_col)
            X_test = pd.DataFrame(X_scaler.transform(X_test), columns=fs_col)
            y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)
            m = copy.deepcopy(model)
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            y_pred.extend(pred)
            y_true.extend(y_test)
            y_pred_inv.extend(y_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1))
            y_true_inv.extend(y_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1))
            try:
                fi_ar += np.abs(m.feature_importances_)
            except:
                fi_ar += np.abs(m.coef_.reshape(-1))
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        y_pred_inv, y_true_inv = np.array(y_pred_inv), np.array(y_true_inv)
        gt_pred.insert(0, [y_true, y_pred])
        gt_pred_inv.insert(0, [y_true_inv, y_pred_inv])
        r2 = np.round(r2_score(y_true, y_pred), 3)
        adj_r2 = np.round(1 - (1-r2) * (len(y_true)-1) / (len(y_true) - X.shape[1] - 1), 3)
        mse = np.round(mean_squared_error(y_true, y_pred), 3)
        mae = np.round(mean_absolute_error(y_true, y_pred), 3)
        rmsle = np.round(root_mean_squared_log_error(y_true_inv, y_pred_inv), 3)
        pcc, pval = scipy.stats.pearsonr(y_true, y_pred)
        pcc = np.round(pcc, 3)
        mse_list.insert(0, mse)
        mae_list.insert(0, mae)
        rmsle_list.insert(0, rmsle)
        r2_list.insert(0, r2)
        adj_r2_list.insert(0, adj_r2)
        pcc_list.insert(0, pcc)
        pval_list.insert(0, pval)
        print('Num of features: {} | R2: {}, PCC: {}'.format(fs_col.shape[0], r2, pcc))
        fi = fi_ar / k
        if fi.shape[0] == 1:
            break
        if first:
            sort_fi = sorted(dict(zip(fs_col, fi)).items(), key=lambda x: x[1], reverse=True)[:n]
            fi_all = np.array(sort_fi)
            first = False
        else:
            sort_fi = sort_fi[:-1]
        fs_col = np.array([j[0] for j in sort_fi])
    return np.array(mse_list), np.array(mae_list), np.array(rmsle_list), np.array(r2_list), np.array(adj_r2_list), np.array(pcc_list), np.array(pval_list), np.array(feat_list, dtype=object), np.array(gt_pred), np.array(gt_pred_inv), fi_all
