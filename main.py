import pandas as pd
import numpy as np
from utils import load_data, load_fps, seed_everything
from trainer import kfold_cv_thres_fps
from sklearn.ensemble import RandomForestRegressor

seed = 42
seed_everything(seed)

data_path = 'experiments datapath' # fill
fps_path = 'fingerprints datapath' # fill
feat, y = load_data(data_path)
fps = load_fps(fps_path)
X = pd.concat([feat, fps], axis=1)
model = RandomForestRegressor(random_state=seed)
mse_list, mae_list, rmsle_list, r2_list, adj_r2_list, pcc_list, pval_list, feat_list, gt_pred, gt_pred_inv, fi_all = kfold_cv_thres_fps(X, y, model, 10, n=50)
best_idx = np.argmax(r2_list)
metrics = '{},{},{},{},{},{}'.format(mse_list[best_idx], mae_list[best_idx], r2_list[best_idx], pcc_list[best_idx], pval_list[best_idx], feat_list[best_idx])
print("Best RMSE:", mse_list[best_idx])
print("Best MAE:", mae_list[best_idx])
print("Best RMSLE", rmsle_list[best_idx])
print("Best R2:", r2_list[best_idx])
print("Best Adj-R2:", adj_r2_list[best_idx])
print("Best PCC:", pcc_list[best_idx])
print('Best p-value', pval_list[best_idx])
print("Best Feature sets:", feat_list[best_idx])