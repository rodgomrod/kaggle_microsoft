import numpy as np
import pandas as pd
import gc
from catboost import CatBoostClassifier
from sklearn.externals import joblib

# TRAIN "RAW"
path = 'data/test.csv'

save_feature_importances = 1

# Para full test quitar el nrows
test = pd.read_csv(path, low_memory=True, nrows=100000)
test = test.fillna(0)

sel_cols = [c for c in test.columns if c not in ['MachineIdentifier',
                                                  'HasDetections']]

X_test = test.loc[:, sel_cols]
X_machines = test.loc[:,'MachineIdentifier']
del test
gc.collect()

print('Cargando Modelo')
model = joblib.load('saved_models/catboost_raw.pkl')

print('Realizando y guardando predicciones')
preds = model.predict_proba(X_test)
preds_1 = preds[:,1]

df_prds = pd.DataFrame({'MachineIdentifier': X_machines, 'HasDetections': preds_1})

df_prds.to_csv('submissions/catboost_raw.csv', index=None)