import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
import lightgbm as lgb
import gc
import seaborn as sns
import matplotlib.pyplot as plt

from utils.schemas import schema_train_4

#Save importances:
save_feature_importances = 1

#Dict para reducir memoria:

print('Cargando datos del TRAIN')
path = 'data/train_final_4'
allFiles = glob.glob(path + "/*.csv")
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_)
    df = (df.fillna(-1)).astype(schema_train_4)
    list_.append(df)

train = pd.concat(list_, axis = 0, ignore_index = True)

drop_version = ['AvSigVersion_index', 'EngineVersion_index', 'Census_OSVersion_index', 'AppVersion_index']
sel_cols = [c for c in train.columns if c not in ['MachineIdentifier',
                                                      'HasDetections',
                                                      'Census_DeviceFamily_Windows.Server',
                                                      'Census_DeviceFamily_Windows.Desktop'
                                                     ]+drop_version]

X_train = train.loc[:, sel_cols]
y_train = train.loc[:,'HasDetections']
del train
del list_
gc.collect()

train_ids = X_train.index
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf.get_n_splits(train_ids, y_train)

# Bajar max_depth para evitar overfitting
print('Comienza entrenamiento del modelo LightGBM')
lgb_model = lgb.LGBMClassifier(max_depth=10,
                               n_estimators=10000,
                               learning_rate=0.05,
                               num_leaves=256,
                               colsample_bytree=0.2,
                               objective='binary',
                               n_jobs=-1)

counter = 1
for train_index, test_index in skf.split(train_ids, y_train):
    print('Fold {}\n'.format(counter))

    X_fit, X_val = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
    y_fit, y_val = y_train.iloc[train_index], y_train.iloc[test_index]

    lgb_model.fit(X_fit, y_fit,
                  eval_set=[(X_val, y_val)],
                  verbose=50, early_stopping_rounds=15)

    del X_fit, X_val, y_fit, y_val, train_index, test_index
    gc.collect()

    print('Guardamos el modelo')
    joblib.dump(lgb_model, 'saved_models/lgbc_model_6_{}.pkl'.format(counter))

    counter += 1


print('Guardamos el modelo')
joblib.dump(lgb_model, 'saved_models/lgbc_model_6.pkl')
del X_train
del y_train
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
    plt.savefig('importances/lgbc_model_6_importances.png')

    importance_df.sort_values(by="importance", ascending=False) \
        .to_csv('importances/feature_importances_lgbc_model_6.csv', index=True)

