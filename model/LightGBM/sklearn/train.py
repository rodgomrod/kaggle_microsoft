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
import sys


save_feature_importances = 1

params = eval(sys.argv[1])
k = int(sys.argv[2])
drop_version = int(sys.argv[3])
model_name = sys.argv[4]
ftimp = int(sys.argv[5])

print('Cargando datos del TRAIN')
path = 'data/train_final_4'
allFiles = glob.glob(path + "/*.csv")
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_)
    df = (df.fillna(-1)).astype(schema_train_4)
    list_.append(df)

train = pd.concat(list_, axis = 0, ignore_index = True)

if drop_version:
    drop_version = ['AvSigVersion_index', 'EngineVersion_index', 'Census_OSVersion_index', 'AppVersion_index']
else:
    drop_version = []

sel_cols = [c for c in train.columns if c not in ['MachineIdentifier',
                                                      'HasDetections',
                                                      'Census_DeviceFamily_Windows.Server',
                                                      'Census_DeviceFamily_Windows.Desktop'
                                                     ]+drop_version]

if ftimp:
    sel_cols = ['AVProductStatesIdentifier',
                 'CountryIdentifier',
                 'CityIdentifier',
                 'max_AvSigVersion_diff',
                 'Census_ProcessorModelIdentifier',
                 'Census_SystemVolumeTotalCapacity',
                 'count8',
                 'count5',
                 'count7',
                 'Census_FirmwareVersionIdentifier',
                 'LocaleEnglishNameIdentifier',
                 'Census_OEMModelIdentifier',
                 'Census_OSBuildRevision',
                 'count1',
                 'AvSigVersion_1_index',
                 'Census_InternalPrimaryDiagonalDisplaySizeInInches',
                 'Wdft_RegionIdentifier',
                 'GeoNameIdentifier',
                 'SmartScreen_index',
                 'count4',
                 'count(DISTINCT AvSigVersion_Name)',
                 'max_OSVersion_diff',
                 'Census_OSInstallTypeName_index',
                 'count6',
                 'Census_OEMNameIdentifier',
                 'prediction_64',
                 'IeVerIdentifier',
                 'OsBuildLab_index',
                 'Census_ActivationChannel_index',
                 'Census_PrimaryDiskTotalCapacity',
                 'AppVersion_1_index',
                 'Census_InternalBatteryNumberOfCharges',
                 'AppVersion_0_index',
                 'Census_TotalPhysicalRAM',
                 'max_OsBuildLab_diff',
                 'Census_OSInstallLanguageIdentifier',
                 'Census_OSUILocaleIdentifier',
                 'Census_FirmwareManufacturerIdentifier',
                 'count2',
                 'Wdft_IsGamer']

X_train = train.loc[:, sel_cols]
y_train = train.loc[:,'HasDetections']
del train
del list_
gc.collect()

train_ids = X_train.index
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
skf.get_n_splits(train_ids, y_train)

print('Comienza entrenamiento del modelo LightGBM')
lgb_model = lgb.LGBMClassifier(**params)

ft_importances = np.zeros(X_train.shape[1])

counter = 1
for train_index, test_index in skf.split(train_ids, y_train):
    print('Fold {}\n'.format(counter))

    X_fit, X_val = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
    y_fit, y_val = y_train.iloc[train_index], y_train.iloc[test_index]

    lgb_model.fit(X_fit,
                  y_fit,
                  eval_set=[(X_val, y_val)],
                  verbose=100,
                  early_stopping_rounds=100)

    del X_fit, X_val, y_fit, y_val, train_index, test_index
    gc.collect()

    print('Guardamos el modelo')
    joblib.dump(lgb_model, 'saved_models/{}_{}.pkl'.format(model_name, counter))

    ft_importances += lgb_model.feature_importances_

    counter += 1

columnas = X_train.columns
del X_train
del y_train
gc.collect()

if save_feature_importances:
    imp = pd.DataFrame({'feature': columnas, 'importance': ft_importances/k})
    df_imp_sort = imp.sort_values('importance', ascending=False)
    df_imp_sort.to_csv('importances/feature_importances_{}.csv'.format(model_name), index=False)

    plt.figure(figsize=(14, 25))
    sns.barplot(x="importance",
                y="feature",
                data=df_imp_sort)
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig('importances/{}_importances.png'.format(model_name))

