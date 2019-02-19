#!/usr/bin/env bash

echo "Generando DFs con vars categoricas y numericas"
python3 ETL/Extract/numerical_categoric_dataset.py
echo "-------------------"




echo -e "\e[92m\e[5mPreprocesamos variables categoricas\e[0m"
python3 ETL/Preprocessing/categorical.py
echo "-------------------"

rm -rf data/df_cat/



echo -e "\e[92m\e[5mProcesamos variables categoricas\e[0m"
python3 ETL/Transform/categorical/global.py
echo "-------------------"

rm -rf data/df_cat_prepro_0/


echo -e "\e[92m\e[5mImputamos valores numericos\e[0m"
python3 ETL/Transform/numeric/impute_numerical.py
echo "-------------------"

rm -rf data/df_num/


echo -e "\e[92m\e[5mGeneramos TRAIN / TEST de las variables nuevas\e[0m"
python3 ETL/Load/train_test_new_variables.py
echo "-------------------"

rm -rf  data/df_cat_pro_0/
rm -rf  data/df_num_imputed_0/