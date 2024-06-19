import random
import numpy as np
import os

import pandas as pd

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_data(path):
    feat_col = ['NP', 'Mass', 'Lipid', 'DSPC', 'DOPE', 'Chol', 'PEG']
    gt_col = ['Exp1', 'Exp2', 'Exp3', 'Exp4']
    df = pd.read_csv(path, encoding='utf-8', header=0)
    df = df.drop(df[(df['Opt'] == 5) | (df['Opt'] == 6)].index, axis=0)
    df = df.drop(df[df['Num'] == 74].index, axis=0)
    df = df.reset_index(drop=True)
    return df[feat_col], np.log(df[gt_col].mean(axis=1).values+1)

def load_fps(path):
    df = pd.read_csv(path, index_col=0)
    df = df.reset_index(drop=True)
    return df