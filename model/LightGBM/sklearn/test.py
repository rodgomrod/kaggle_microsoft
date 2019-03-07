import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
import lightgbm as lgb
import xgboost as xgb
from scipy.sparse import vstack, csr_matrix, save_npz, load_npz
import warnings
warnings.filterwarnings("ignore")
import gc

from utils.schemas import dict_dtypes_onehot_schema, schema_train_3, schema_test_4


print('Cargando datos del TEST')
path = 'data/test_final_4'
allFiles = glob.glob(path + "/*.csv")
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_)
    df = (df.fillna(-1)).astype(schema_test_4)
    list_.append(df)


test = pd.concat(list_, axis = 0, ignore_index = True).fillna(-1)

sel_cols = [c for c in test.columns if c not in ['MachineIdentifier',
                                                 'HasDetections',
                                                 'Census_DeviceFamily_Windows.Server',
                                                 'Census_DeviceFamily_Windows.Desktop'
                                                 ]
            ]

X_test = test.loc[:, sel_cols]
X_machines = test.loc[:, 'MachineIdentifier']
del test
del list_
gc.collect()

print('Cargando Modelo')
model = joblib.load('saved_models/lgbc_model_5.pkl')

print('Realizando y guardando predicciones')
preds = model.predict_proba(X_test)
preds_1 = preds[:,1]

df_prds = pd.DataFrame({'MachineIdentifier': X_machines, 'HasDetections': preds_1})

df_prds.to_csv('submissions/lgbc_model_5.csv', index=None)
