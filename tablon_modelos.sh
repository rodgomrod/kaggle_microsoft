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


params="{'min_data_in_leaf':20,'max_depth':-1,'metric':'auc','n_estimators':10000,'learning_rate':0.05,'num_leaves':75,'colsample_bytree':1,'objective':'binary','n_jobs':-1,'seed':42,'bagging_fraction':1,'lambda_l1':0,'lambda_l2':0}"

k="3"

drop_version="1"

model_name="lgbc_model_15"

ftsel="0"


echo -e "\e[92m\e[5mEntrenamos modelo de LightGBM \e[34m${model_name}\e[0m"
python3 model/LightGBM/sklearn/train.py ${params} ${k} ${drop_version} ${model_name} ${ftsel}
echo ""

echo -e "\e[92m\e[5mPredicciones de LightGBM \e[34m${model_name}\e[0m"
python3 model/LightGBM/sklearn/test.py ${k} ${drop_version} ${model_name} ${ftsel}
echo ""

echo -e "\e[92m\e[5mSubmitting predictions \e[34m${model_name}\e[0m"
kaggle competitions submit -c microsoft-malware-prediction -f submissions/${model_name}.csv -m "Nuevos datos V4: con versiones - ${params} - k = ${k} Versions = ${drop_version} y ftsel: ${ftsel}. Con Cat_cols con fillna = 0"
echo ""
echo ""
echo ""



