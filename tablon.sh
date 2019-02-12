#!/usr/bin/env bash

#echo "Generando DFs con vars categoricas y numericas"
#python3 ETL/Extract/numerical_categoric_dataset.py
#echo ""

echo -e "\e[92m\e[5mPreprocesamos variables categoricas\e[0m"
#python3 ETL/Preprocessing/categorical.py
spark-submit --master local[*] ETL/Preprocessing/categorical.py
echo ""

echo -e "\e[92m\e[5mProcesamos variables categoricas\e[0m"
#python3 ETL/Transform/categorical/global.py
spark-submit --master local[*] ETL/Transform/categorical/global.py
echo ""

#echo -e "\e[92m\e[5mImputamos valores numericos\e[0m"
##python3 ETL/Transform/numeric/impute_numerical.py
#spark-submit --master local[*] ETL/Transform/numeric/impute_numerical.py
#echo ""

echo -e "\e[92m\e[5mGeneramos TRAIN / TEST de las variables nuevas\e[0m"
#python3 ETL/Load/train_test_new_variables.py
spark-submit --master local[*] ETL/Load/train_test_new_variables.py
echo ""
