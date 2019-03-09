#!/usr/bin/env bash


params="{'max_depth':10,'n_estimators':10000,'learning_rate':0.07,'num_leaves':31,'colsample_bytree':0.2,'objective':'binary','n_jobs':-1,'seed':42,'bagging_fraction':0.5,'bagging_freq':10,'lambda_l1':0.2,'lambda_l2':0.2}"

k="3"

drop_version="1"

model_name="lgbc_model_7"


echo -e "\e[92m\e[5mEntrenamos modelo de LightGBM \e[34m${model_name}\e[0m"
python3 model/LightGBM/sklearn/train.py ${params} ${k} ${drop_version} ${model_name}
echo ""

echo -e "\e[92m\e[5mPredicciones de LightGBM \e[34m${model_name}\e[0m"
python3 model/LightGBM/sklearn/test.py ${k} ${drop_version} ${model_name}
echo ""

echo -e "\e[92m\e[5mSubmitting predictions \e[34m${model_name}\e[0m"
kaggle competitions submit -c microsoft-malware-prediction -f submissions/${model_name}.csv -m "Nuevos datos V4:
 sin versiones - ${params}"
echo ""
echo ""
echo ""

params="{'max_depth':13,'n_estimators':10000,'learning_rate':0.09,'num_leaves':31,'colsample_bytree':0.2,'objective':'binary','n_jobs':-1,'seed':42}"

k="3"

drop_version="1"

model_name="lgbc_model_8"


echo -e "\e[92m\e[5mEntrenamos modelo de LightGBM \e[34m${model_name}\e[0m"
python3 model/LightGBM/sklearn/train.py ${params} ${k} ${drop_version} ${model_name}
echo ""

echo -e "\e[92m\e[5mPredicciones de LightGBM \e[34m${model_name}\e[0m"
python3 model/LightGBM/sklearn/test.py ${k} ${drop_version} ${model_name}
echo ""

echo -e "\e[92m\e[5mSubmitting predictions \e[34m${model_name}\e[0m"
kaggle competitions submit -c microsoft-malware-prediction -f submissions/${model_name}.csv -m "Nuevos datos V4: sin versiones - ${params}"
echo ""
echo ""
echo ""

params="{'max_depth':10,'n_estimators':10000,'learning_rate':0.09,'num_leaves':31,'colsample_bytree':0.2,'objective':'binary','n_jobs':-1,'seed':42,'bagging_fraction':0.5,'bagging_freq':10,'lambda_l1':0.1,'lambda_l2':0.1}"

k="3"

drop_version="0"

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
