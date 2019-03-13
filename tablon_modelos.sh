#!/usr/bin/env bash


#params="{'min_data_in_leaf':20,'max_depth':9,'metric':'auc','n_estimators':10000,'learning_rate':0.3,'num_leaves':75,'colsample_bytree':1,'objective':'binary','n_jobs':-1,'seed':42,'bagging_fraction':1,'lambda_l1':0,'lambda_l2':0}"
#
#k="5"
#
#drop_version="0"
#
#model_name="lgbc_model_11"
#
#
#echo -e "\e[92m\e[5mEntrenamos modelo de LightGBM \e[34m${model_name}\e[0m"
#python3 model/LightGBM/sklearn/train.py ${params} ${k} ${drop_version} ${model_name}
#echo ""
#
#echo -e "\e[92m\e[5mPredicciones de LightGBM \e[34m${model_name}\e[0m"
#python3 model/LightGBM/sklearn/test.py ${k} ${drop_version} ${model_name}
#echo ""
#
#echo -e "\e[92m\e[5mSubmitting predictions \e[34m${model_name}\e[0m"
#kaggle competitions submit -c microsoft-malware-prediction -f submissions/${model_name}.csv -m "Nuevos datos V4: con versiones - ${params} - k = ${k} Versions = ${drop_version}"
#echo ""
#echo ""
#echo ""
#
#params="{'min_data_in_leaf':20,'max_depth':7,'metric':'auc','n_estimators':10000,'learning_rate':0.5,'num_leaves':75,'colsample_bytree':1,'objective':'binary','n_jobs':-1,'seed':42,'bagging_fraction':0.9,'lambda_l1':0,'lambda_l2':0}"
#
#k="7"
#
#drop_version="1"
#
#model_name="lgbc_model_12"
#
#
#echo -e "\e[92m\e[5mEntrenamos modelo de LightGBM \e[34m${model_name}\e[0m"
#python3 model/LightGBM/sklearn/train.py ${params} ${k} ${drop_version} ${model_name}
#echo ""
#
#echo -e "\e[92m\e[5mPredicciones de LightGBM \e[34m${model_name}\e[0m"
#python3 model/LightGBM/sklearn/test.py ${k} ${drop_version} ${model_name}
#echo ""
#
#echo -e "\e[92m\e[5mSubmitting predictions \e[34m${model_name}\e[0m"
#kaggle competitions submit -c microsoft-malware-prediction -f submissions/${model_name}.csv -m "Nuevos datos V4: con versiones - ${params} - k = ${k} Versions = ${drop_version}"
#echo ""
#echo ""
#echo ""


#params="{'min_data_in_leaf':20,'max_depth':-1,'metric':'auc','n_estimators':10000,'learning_rate':0.05,'num_leaves':75,'colsample_bytree':1,'objective':'binary','n_jobs':-1,'seed':42,'bagging_fraction':1,'lambda_l1':0,'lambda_l2':0}"
#
#k="3"
#
#drop_version="1"
#
#model_name="lgbc_model_15"
#
#ftsel="0"
#
#
#echo -e "\e[92m\e[5mEntrenamos modelo de LightGBM \e[34m${model_name}\e[0m"
#python3 model/LightGBM/sklearn/train.py ${params} ${k} ${drop_version} ${model_name} ${ftsel}
#echo ""
#
#echo -e "\e[92m\e[5mPredicciones de LightGBM \e[34m${model_name}\e[0m"
#python3 model/LightGBM/sklearn/test.py ${k} ${drop_version} ${model_name} ${ftsel}
#echo ""
#
#echo -e "\e[92m\e[5mSubmitting predictions \e[34m${model_name}\e[0m"
#kaggle competitions submit -c microsoft-malware-prediction -f submissions/${model_name}.csv -m "Nuevos datos V4: con versiones - ${params} - k = ${k} Versions = ${drop_version} y ftsel: ${ftsel}. Con Cat_cols con fillna = 0"
#echo ""
#echo ""
#echo ""


#params="{'max_depth':7,'n_estimators':1000,'colsample_bytree':0.2,'learning_rate':0.1,'objective':'binary:logistic','n_jobs':-1,'eval_metric':'auc'}"
#
#k="3"
#
#drop_version="1"
#
#model_name="xgb_model_1"
#
#ftsel="0"
#
#
#echo -e "\e[92m\e[5mEntrenamos modelo de XGBoost \e[34m${model_name}\e[0m"
#python3 model/XGBoost/sklearn/train.py ${params} ${k} ${drop_version} ${model_name} ${ftsel}
#echo ""
#
#echo -e "\e[92m\e[5mPredicciones de XGBoost \e[34m${model_name}\e[0m"
#python3 model/XGBoost/sklearn/test.py ${k} ${drop_version} ${model_name} ${ftsel}
#echo ""
#
#echo -e "\e[92m\e[5mSubmitting predictions \e[34m${model_name}\e[0m"
#kaggle competitions submit -c microsoft-malware-prediction -f submissions/${model_name}.csv -m "Nuevos datos V4 (XGB): con versiones - ${params} - k = ${k} Versions = ${drop_version} y ftsel: ${ftsel}."
#echo ""
#echo ""
#echo ""

params="{'depth':11,'iterations':1000,'eval_metric':'AUC','random_seed':42,'logging_level':'Verbose','allow_writing_files':False,'early_stopping_rounds':50,'learning_rate':0.5,'thread_count':8,'boosting_type':'Plain','bootstrap_type':'Bernoulli','rsm':0.3}"

k="2"

drop_version="1"

model_name="catboost_model_2"

ftsel="0"


echo -e "\e[92m\e[5mEntrenamos modelo de CatBoost \e[34m${model_name}\e[0m"
python3 model/CatBoost/train.py ${params} ${k} ${drop_version} ${model_name} ${ftsel}
echo ""

echo -e "\e[92m\e[5mPredicciones de CatBoost \e[34m${model_name}\e[0m"
python3 model/CatBoost/test.py ${k} ${drop_version} ${model_name} ${ftsel}
echo ""

echo -e "\e[92m\e[5mSubmitting predictions \e[34m${model_name}\e[0m"
kaggle competitions submit -c microsoft-malware-prediction -f submissions/${model_name}.csv -m "Nuevos datos V4 (CaBoost): con versiones - ${params} - k = ${k} Versions = ${drop_version} y ftsel: ${ftsel}."
echo ""
echo ""
echo ""



