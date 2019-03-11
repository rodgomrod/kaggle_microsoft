import pandas as pd
import numpy as np
from sklearn.externals import joblib
import gc
from utils.schemas import schema_train_4, schema_test_4
import glob

print('Cargando train')
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

print('Guardando train')
np.save('data/npy/train_ids.npy', train['MachineIdentifier'])
np.save('data/npy/train_dataset.npy', train[sel_cols])
np.save('data/npy/train_target.npy', train['HasDetections'])

del train
gc.collect()

print('Cargando test')
path = 'data/test_final_4'
allFiles = glob.glob(path + "/*.csv")
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_)
    df = (df.fillna(-1)).astype(schema_test_4)
    list_.append(df)

test = pd.concat(list_, axis = 0, ignore_index = True)

print('Guardando test')
np.save('data/npy/test_ids.npy', test['MachineIdentifier'])
np.save('data/npy/test_dataset.npy', test[sel_cols])

# y = np.load('../datasets/model_32_train_target.npy')

