import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import TimeSeriesSplit
from sklearn.externals import joblib
import lightgbm as lgb
import gc
import seaborn as sns
import matplotlib.pyplot as plt

save_feature_importances = 1

from utils.schemas import dict_dtypes_onehot_schema


print('Cargando datos del TRAIN')
path = 'data/train_final_2'
allFiles = glob.glob(path + "/*.csv")
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_, dtype=dict_dtypes_onehot_schema, low_memory=True)
    list_.append(df)

train = pd.concat(list_, axis = 0, ignore_index = True)
train = train.sort_values(['AvSigVersion_0',
                           'AvSigVersion_1',
                           'AvSigVersion_2',
                           'OsVer_0',
                           'OsVer_2',
                           'OsVer_3',], ascending=True)

sel_cols = [c for c in train.columns if c not in ['MachineIdentifier',
                                                      'HasDetections',
                                                      'Census_DeviceFamily_Windows.Server',
                                                      'Census_DeviceFamily_Windows.Desktop'
                                                     ]]

X = train.loc[:, sel_cols]
y = train.loc[:, 'HasDetections']
del train
del list_
gc.collect()


# Bajar max_depth para evitar overfitting
print('Comienza entrenamiento del modelo LightGBM')
lgb_model = lgb.LGBMClassifier(max_depth=15,
                               n_estimators=10000,
                               learning_rate=0.05,
                               num_leaves=256,
                               colsample_bytree=0.27,
                               objective='binary',
                               lambda_l1=0.1,
                               lambda_l2=0.1,
                               n_jobs=-1)

tscv = TimeSeriesSplit(n_splits=9)
n = 1
for train_index, test_index in tscv.split(X=X, y=y):
    print('FOLD {}'.format(n))
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    lgb_model.fit(X_train, y_train, eval_metric='auc',
                  eval_set=[(X_test, y_test)],
                  verbose=50, early_stopping_rounds=20)
    print(train_index, test_index)
    del X_train
    del X_test
    del y_train
    del y_test
    gc.collect()
    n += 1

print('Best score en el entrenamiento:', lgb_model.best_score_)

print('Guardamos el modelo')
joblib.dump(lgb_model, 'saved_models/lgbc_tss_0.pkl')
del X
del y
gc.collect()

if save_feature_importances:
    importance_df = pd.DataFrame()
    importance_df["importance"] = lgb_model.feature_importances_()
    importance_df["feature"] = sel_cols

    plt.figure(figsize=(14, 25))
    sns.barplot(x="importance",
                y="feature",
                data=importance_df.sort_values(by="importance",
                                               ascending=False))
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig('importances/lgbc_model_3_importances.png')

    importance_df.sort_values(by="importance", ascending=False) \
        .to_csv('importances/feature_importances_lgbc_model_3.csv', index=True)




