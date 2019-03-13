import pandas as pd
import numpy as np
import glob
from sklearn.externals import joblib
import lightgbm as lgb
import gc
from utils.schemas import schema_test_4
import sys

k = int(sys.argv[1])
drop_version = int(sys.argv[2])
model_name = sys.argv[3]
ftimp = int(sys.argv[4])

print('Cargando datos del TEST')
path = 'data/test_final_4'
allFiles = glob.glob(path + "/*.csv")
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_)
    df = (df.fillna(-1)).astype(schema_test_4)
    list_.append(df)


test = pd.concat(list_, axis = 0, ignore_index = True)

if drop_version:
    drop_version = ['AvSigVersion_index', 'EngineVersion_index', 'Census_OSVersion_index', 'AppVersion_index']
else:
    drop_version = []

sel_cols = [c for c in test.columns if c not in ['MachineIdentifier',
                                                 'HasDetections',
                                                 'Census_DeviceFamily_Windows.Server',
                                                 'Census_DeviceFamily_Windows.Desktop'
                                                 ]+drop_version
            ]

X_test = test.loc[:, sel_cols]
X_machines = test.loc[:, 'MachineIdentifier']
del test
del list_
gc.collect()

cb_test_preds = np.zeros(X_test.shape[0])

for i in range(1, k+1):
    model = joblib.load('saved_models/{}_{}.pkl'.format(model_name, i))
    print('Realizando predicciones. FOLD = {}'.format(i))
    cb_test_preds += model.predict_proba(X_test)[:, 1]

    del model
    gc.collect()

del X_test
gc.collect()

print('Haciendo la media y guardando CSV')
final_prds = cb_test_preds/k

df_prds = pd.DataFrame({'MachineIdentifier': X_machines, 'HasDetections': final_prds})

df_prds.to_csv('submissions/{}.csv'.format(model_name), index=None)
