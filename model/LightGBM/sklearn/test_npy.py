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

print('Cargando Modelos')

model1 = joblib.load('saved_models/lgbc_model_6_1.pkl')
model2 = joblib.load('saved_models/lgbc_model_6_2.pkl')
model3 = joblib.load('saved_models/lgbc_model_6_3.pkl')
model4 = joblib.load('saved_models/lgbc_model_6_4.pkl')
model5 = joblib.load('saved_models/lgbc_model_6_5.pkl')

print('Realizando y guardando predicciones')
preds1 = model1.predict_proba(X_test)
preds_1 = preds1[:,1]
preds2 = model2.predict_proba(X_test)
preds_2 = preds2[:,1]
preds3 = model3.predict_proba(X_test)
preds_3 = preds3[:,1]
preds4 = model4.predict_proba(X_test)
preds_4 = preds4[:,1]
preds5 = model5.predict_proba(X_test)
preds_5 = preds5[:,1]

del X_test
gc.collect()

final_prds = (preds1 + preds2 + preds3 + preds4 + preds5)/5

df_prds = pd.DataFrame({'MachineIdentifier': X_machines, 'HasDetections': final_prds})

df_prds.to_csv('submissions/lgbc_model_6.csv', index=None)
