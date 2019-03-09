import pandas as pd
import numpy as np
import glob
from sklearn.externals import joblib
import lightgbm as lgb
import gc

from utils.schemas import schema_test_4


print('Cargando datos del TEST')
path = 'data/test_final_4'
allFiles = glob.glob(path + "/*.csv")
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_)
    df = (df.fillna(-1)).astype(schema_test_4)
    list_.append(df)


test = pd.concat(list_, axis = 0, ignore_index = True).fillna(-1)

drop_version = ['AvSigVersion_index', 'EngineVersion_index', 'Census_OSVersion_index', 'AppVersion_index']
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

print('Cargando Modelos')
model1 = joblib.load('saved_models/lgbc_model_6_1.pkl')
model2 = joblib.load('saved_models/lgbc_model_6_2.pkl')
model3 = joblib.load('saved_models/lgbc_model_6_3.pkl')
model4 = joblib.load('saved_models/lgbc_model_6_4.pkl')
model5 = joblib.load('saved_models/lgbc_model_6_5.pkl')

print('Realizando predicciones 1')
preds1 = model1.predict_proba(X_test)
preds_1 = preds1[:,1]
del preds1
gc.collect()

print('Realizando predicciones 2')
preds2 = model2.predict_proba(X_test)
preds_2 = preds2[:,1]
del preds2
gc.collect()

print('Realizando predicciones 3')
preds3 = model3.predict_proba(X_test)
preds_3 = preds3[:,1]
del preds3
gc.collect()

print('Realizando predicciones 4')
preds4 = model4.predict_proba(X_test)
preds_4 = preds4[:,1]
del preds4
gc.collect()

print('Realizando predicciones 5')
preds5 = model5.predict_proba(X_test)
preds_5 = preds5[:,1]
del preds5
del X_test
gc.collect()

print('Haciendo la media y guardando CSV')
final_prds = (preds_1 + preds_2 + preds_3 + preds_4 + preds_5)/5

df_prds = pd.DataFrame({'MachineIdentifier': X_machines, 'HasDetections': final_prds})

df_prds.to_csv('submissions/lgbc_model_6.csv', index=None)
