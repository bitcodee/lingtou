import pandas as pd
import torch

import torchvision

csv_path = './data/'

df = pd.read_csv('./data/tjid_Pre-diabetes_0420.csv')

## with all year data
allYear = df[~df.isnull().T.any()]

allYear.tjid.to_csv('./data/pre_trn_id.csv', index=False)



1