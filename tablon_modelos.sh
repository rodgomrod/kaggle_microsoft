#!/usr/bin/env bash

params="{'min_data_in_leaf':20,'max_depth':50,'metric':'auc','n_estimators':10000,'learning_rate':0.2,'num_leaves':75,'colsample_bytree':1,'objective':'binary','n_jobs':-1,'seed':42,'bagging_fraction':1,'bagging_freq':10,'lambda_l1':0.2,'lambda_l2':0.2}"

k="5"

drop_version="1"

model_name="lgbc_model_9"


echo -e "\e[92m\e[5mEntrenamos modelo de LightGBM \e[34m${model_name}\e[0m"
python3 model/LightGBM/sklearn/train.py ${params} ${k} ${drop_version} ${model_name}
echo ""

echo -e "\e[92m\e[5mPredicciones de LightGBM \e[34m${model_name}\e[0m"
python3 model/LightGBM/sklearn/test.py ${k} ${drop_version} ${model_name}
echo ""

echo -e "\e[92m\e[5mSubmitting predictions \e[34m${model_name}\e[0m"
kaggle competitions submit -c microsoft-malware-prediction -f submissions/${model_name}.csv -m "Nuevos datos V4: con versiones - ${params}"
echo ""
echo ""
echo ""


params="{'min_data_in_leaf':20,'max_depth':-1,'metric':'auc','n_estimators':10000,'learning_rate':0.05,'num_leaves':75,'colsample_bytree':1,'objective':'binary','n_jobs':-1,'seed':42,'bagging_fraction':1,'lambda_l1':0,'lambda_l2':0}"

k="5"

drop_version="1"

model_name="lgbc_model_10"


echo -e "\e[92m\e[5mEntrenamos modelo de LightGBM \e[34m${model_name}\e[0m"
python3 model/LightGBM/sklearn/train.py ${params} ${k} ${drop_version} ${model_name}
echo ""

echo -e "\e[92m\e[5mPredicciones de LightGBM \e[34m${model_name}\e[0m"
python3 model/LightGBM/sklearn/test.py ${k} ${drop_version} ${model_name}
echo ""

echo -e "\e[92m\e[5mSubmitting predictions \e[34m${model_name}\e[0m"
kaggle competitions submit -c microsoft-malware-prediction -f submissions/${model_name}.csv -m "Nuevos datos V4: con versiones - ${params}"
echo ""
echo ""
echo ""
