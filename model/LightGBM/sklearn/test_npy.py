import pandas as pd
import numpy as np
import glob

from sklearn.externals import joblib
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
import gc


X_test = np.load('data/npy/test_dataset.npy')
X_machines = np.load('data/npy/test_ids.npy')

print('Cargando Modelo')
model = joblib.load('saved_models/lgbc_model_6.pkl')

print('Realizando y guardando predicciones')
preds = model.predict_proba(X_test)
preds_1 = preds[:,1]

del X_test
gc.collect()

df_prds = pd.DataFrame({'MachineIdentifier': X_machines, 'HasDetections': preds_1})

df_prds.to_csv('submissions/lgbc_model_6.csv', index=None)
