import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
import gc
import seaborn as sns
import matplotlib.pyplot as plt

#Save importances:
save_feature_importances = 1

#Dict para reducir memoria:

print('Cargando datos del TRAIN')
X_train = np.load('data/npy/train_dataset.npy')
y_train = np.load('data/npy/train_target.npy')

train_ids = list(range(len(X_train)))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf.get_n_splits(train_ids, y_train)

# Bajar max_depth para evitar overfitting
print('Comienza entrenamiento del modelo LightGBM')
lgb_model = lgb.LGBMClassifier(max_depth=10,
                               n_estimators=10000,
                               learning_rate=0.03,
                               num_leaves=256,
                               colsample_bytree=0.22,
                               objective='binary',
                               bagging_fraction=0.7,
                               bagging_freq=5,
                               n_jobs=-1)

counter = 1
for train_index, test_index in skf.split(train_ids, y_train):
    print('Fold {}\n'.format(counter))

    X_fit, X_val = X_train[train_index, :], X_train[test_index, :]
    y_fit, y_val = y_train[train_index], y_train[test_index]

    lgb_model.fit(X_fit, y_fit, eval_metric='auc',
                  eval_set=[(X_val, y_val)],
                  verbose=50, early_stopping_rounds=20)

    del X_fit, X_val, y_fit, y_val, train_index, test_index
    gc.collect()

    print('Guardamos el modelo')
    joblib.dump(lgb_model, 'saved_models/lgbc_model_6_{}.pkl'.format(counter))

    counter += 1

del X_train
del y_train
gc.collect()

if save_feature_importances:
    importance_df = pd.DataFrame()
    importance_df["importance"] = lgb_model.feature_importances_()

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

