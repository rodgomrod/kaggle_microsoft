import numpy as np
import pandas as pd
import gc
from catboost import CatBoostClassifier
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
from utils.schemas import schema_train_4
import sys
import glob


params = eval(sys.argv[1])
k = int(sys.argv[2])
drop_version = int(sys.argv[3])
model_name = sys.argv[4]
ftimp = int(sys.argv[5])

# TRAIN "RAW"
path = 'data/train_final_4'

save_feature_importances = 1
print('Cargando datos del TRAIN')
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

X_train = train.loc[:, sel_cols]
y_train = train.loc[:,'HasDetections']
del train
del list_
gc.collect()

cat_ft = ['ProductName_index',
'EngineVersion_0_index',
'EngineVersion_1_index',
'Census_OSVersion_0_index',
'Census_OSVersion_1_index',
'AppVersion_0_index',
'AppVersion_1_index',
'AvSigVersion_0_index',
'AvSigVersion_1_index',
'OsVer_0_index',
'OsVer_1_index',
'Platform_index',
'Processor_index',
'OsPlatformSubRelease_index',
'OsBuildLab_index',
'SkuEdition_index',
'PuaMode_index',
'SmartScreen_index',
'Census_MDC2FormFactor_index',
'Census_DeviceFamily_index',
'Census_ProcessorClass_index',
'Census_PrimaryDiskTypeName_index',
'Census_ChassisTypeName_index',
'Census_PowerPlatformRoleName_index',
'Census_InternalBatteryType_index',
'Census_OSArchitecture_index',
'Census_OSBranch_index',
'Census_OSEdition_index',
'Census_OSSkuName_index',
'Census_OSInstallTypeName_index',
'Census_OSWUAutoUpdateOptionsName_index',
'Census_GenuineStateName_index',
'Census_ActivationChannel_index',
'Census_FlightRing_index',
'OsVer_index',
'DefaultBrowsersIdentifier',
'AVProductStatesIdentifier',
'CountryIdentifier',
'CityIdentifier',
'OrganizationIdentifier',
'GeoNameIdentifier',
'LocaleEnglishNameIdentifier',
'IeVerIdentifier',
'Census_OEMNameIdentifier',
'Census_OEMModelIdentifier',
'Census_ProcessorManufacturerIdentifier',
'Census_ProcessorModelIdentifier',
'Census_OSInstallLanguageIdentifier',
'Census_OSUILocaleIdentifier',
'Census_FirmwareManufacturerIdentifier',
'Census_FirmwareVersionIdentifier',
'Wdft_RegionIdentifier',
'RtpStateBitfield',
'prediction_2',
'prediction_4',
'prediction_8',
'prediction_16',
'prediction_32',
'prediction_64'
]

cat_ft_id = list()
n = 0
for c in sel_cols:
    if c in cat_ft:
        cat_ft_id.append(n)
    n += 1

train_ids = X_train.index
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
skf.get_n_splits(train_ids, y_train)

model_cb = CatBoostClassifier(**params)

print('Comienza entrenamiento del modelo CatBoost')


counter = 0
for train_index, test_index in skf.split(train_ids, y_train):
    print('Fold {}\n'.format(counter + 1))

    X_fit, X_val = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
    y_fit, y_val = y_train.iloc[train_index], y_train.iloc[test_index]

    model_cb.fit(X_fit,
                 y_fit,
                 cat_features=cat_ft_id,
                 eval_set=(X_val, y_val),
                 verbose=10
                 )

    print('Guardamos el modelo')
    joblib.dump(model_cb, 'saved_models/{}_{}.pkl'.format(model_name, counter))


    del X_fit
    del X_val
    del y_fit
    del y_val
    del train_index
    del test_index
    gc.collect()

    counter += 1

del X_train
del y_train
gc.collect()

# if save_feature_importances:
#     importance_df = pd.DataFrame()
#     importance_df["importance"] = model_cb.feature_importances_()
#     importance_df["feature"] = sel_cols
#
#     plt.figure(figsize=(14, 25))
#     sns.barplot(x="importance",
#                 y="feature",
#                 data=importance_df.sort_values(by="importance",
#                                                ascending=False))
#     plt.title('CatBoost Importance Features')
#     plt.tight_layout()
#     plt.savefig('importances/catboost_imp.png')
#
#     importance_df.sort_values(by="importance", ascending=False) \
#         .to_csv('importances/feature_importances_CatBoost.csv', index=True)
