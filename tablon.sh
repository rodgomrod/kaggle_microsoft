#!/usr/bin/env bash

#echo "Generando DFs con vars categoricas y numericas"
#python3 ETL/Extract/numerical_categoric_dataset.py
#echo ""
#
#echo -e "\e[92m\e[5mPreprocesamos variables categoricas\e[0m"
#python3 ETL/Preprocessing/categorical.py
#echo ""

echo -e "\e[92m\e[5mProcesamos variables categoricas\e[0m"
python3 ETL/Transform/categorical/global.py
echo ""
#
echo -e "\e[92m\e[5mImputamos valores numericos\e[0m"
python3 ETL/Transform/numeric/impute_numerical.py
echo ""
#
#echo -e "\e[92m\e[5mCreamos variables KMeans\e[0m"
#python3 ETL/Transform/numeric/kmeans.py
#echo ""

#echo -e "\e[92m\e[5mProcesamos variables de fechas\e[0m"
#python3 ETL/Transform/dates/dates.py
#echo ""
#
echo -e "\e[92m\e[5mProcesamos variable AvSigVersion extra\e[0m"
python3 ETL/Transform/avsigver_extra_info/avsigversion_extra.py
echo ""

echo -e "\e[92m\e[5mGeneramos TRAIN / TEST de las variables nuevas\e[0m"
python3 ETL/Load/train_test_new_variables.py
echo ""

#echo -e "\e[92m\e[5mEntrenamos modelo de LightGBM\e[0m"
#python3 model/LightGBM/sklearn/train.py
#echo ""
#
#echo -e "\e[92m\e[5mPredicciones de LightGBM\e[0m"
#python3 model/LightGBM/sklearn/test.py
#echo ""
#
#echo -e "\e[92m\e[5mSubmitting predictions\e[0m"
#kaggle competitions submit -c microsoft-malware-prediction -f submissions/lgbc_tss_0.csv -m "prueba TSS lgbm model 0
#max_depth=15, n_estimators=10000, learning_rate=0.05, num_leaves=256, colsample_bytree=0.27, objective='binary',
#lambda_l1=0.1, lambda_l2=0.1, n_jobs=-1 k = 9"
#echo ""

#echo -e "\e[92m\e[5mEntrenamos modelo de CatBoost\e[0m"
#python3 model/CatBoost/train.py
#echo ""
#
#echo -e "\e[92m\e[5mPredicciones de CatBoost\e[0m"
#python3 model/CatBoost/test.py
#echo ""

#echo -e "\e[92m\e[5mSubmitting predictions\e[0m"
#kaggle competitions submit -c microsoft-malware-prediction -f submissions/catboost_raw.csv -m "CatBoost raw depth=9,  iterations=600,
#    eval_metric='AUC',
#    random_seed=42,
#    logging_level='Verbose',
#    allow_writing_files=False,
#    metric_period=50,
#    early_stopping_rounds=20,
#    learning_rate=0.1,
#    bagging_temperature=0.9"
#echo ""
