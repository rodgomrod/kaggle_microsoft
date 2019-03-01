import numpy as np
import pandas as pd
import gc
from catboost import CatBoostClassifier
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt

# TRAIN "RAW"
path = 'data/train.csv'

save_feature_importances = 1

print('Cargando datos del TRAIN')
# Para full train quitar el nrows
train = pd.read_csv(path, low_memory=True, nrows=100000)
train = train.fillna(0)

sel_cols = [c for c in train.columns if c not in ['MachineIdentifier',
                                                  'HasDetections']]

X_train = train.loc[:, sel_cols]
y_train = train.loc[:,'HasDetections']
del train
gc.collect()

cat_cols = list()
n = 0
for c, t in zip(sel_cols, X_train.dtypes):
    if t == 'object':
        cat_cols.append(n)
    n+=1

train_ids = X_train.index
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf.get_n_splits(train_ids, y_train)

model_cb = CatBoostClassifier(
    depth=9,
    iterations=1000,
    eval_metric='AUC',
    random_seed=42,
    logging_level='Verbose',
    allow_writing_files=False,
    metric_period=50,
    early_stopping_rounds=20,
    learning_rate=0.1,
    bagging_temperature=0.9
)

print('Comienza entrenamiento del modelo CatBoost')

counter = 0
for train_index, test_index in skf.split(train_ids, y_train):
    print('Fold {}\n'.format(counter + 1))

    X_fit, X_val = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
    y_fit, y_val = y_train.iloc[train_index], y_train.iloc[test_index]

    model_cb.fit(X_fit,
                 y_fit,
                 cat_features=cat_cols,
                 eval_set=(X_val, y_val),
                 # plot=True
                 )

    del X_fit, X_val, y_fit, y_val, train_index, test_index
    gc.collect()

    counter += 1


print('Best score en el entrenamiento:', model_cb.best_score_)

print('Guardamos el modelo')
joblib.dump(model_cb, 'saved_models/catboost_raw.pkl')
del X_train
del y_train
gc.collect()

if save_feature_importances:
    importance_df = pd.DataFrame()
    importance_df["importance"] = model_cb.feature_importances_()
    importance_df["feature"] = sel_cols

    plt.figure(figsize=(14, 25))
    sns.barplot(x="importance",
                y="feature",
                data=importance_df.sort_values(by="importance",
                                               ascending=False))
    plt.title('CatBoost Importance Features')
    plt.tight_layout()
    plt.savefig('importances/catboost_imp.png')

    importance_df.sort_values(by="importance", ascending=False) \
        .to_csv('importances/feature_importances_CatBoost.csv', index=True)