#!/usr/bin/env bash
#clear
#echo "1/7"
echo -e "\e[37;1m\e[5mGenerando DFs con vars categoricas y numericas\e[0m"
python3 ETL/Extract/numerical_categoric_dataset.py
#echo "-------------------"



#clear
#echo "2/7"
#echo -e "\e[37;1m\e[5mPreprocesamos variables categoricas\e[0m"
#python3 ETL/Preprocessing/categorical.py
#echo "-------------------"
#
## rm -rf data/df_cat/
#
#
#clear
#echo "3/7"
#echo -e "\e[37;1m\e[5mProcesamos variables categoricas\e[0m"
#python3 ETL/Transform/categorical/global.py
#echo "-------------------"
#
## rm -rf data/df_cat_prepro_0/
#
#clear
#echo "4/7"
#echo -e "\e[37;1m\e[5mImputamos valores numericos\e[0m"
#python3 ETL/Transform/numeric/impute_numerical.py
#echo "-------------------"
#
## rm -rf data/df_num/
#
#clear
#echo "5/7"
#echo -e "\e[92m\e[5mCreamos variables KMeans\e[0m"
#python3 ETL/Transform/numeric/kmeans.py
#echo ""
#echo "-------------------"
#
#
#clear
echo "6/7"
echo -e "\e[37;1m\e[5mProcesamos variables de fechas\e[0m"
python3 ETL/Transform/dates/dates.py
echo ""


#clear
#echo "7/7"
#echo -e "\e[37;1m\e[5mGeneramos TRAIN / TEST de las variables nuevas\e[0m"
#python3 ETL/Load/train_test_new_variables.py
#echo "-------------------"

# rm -rf  data/df_cat_pro_0/
# rm -rf  data/df_num_imputed_0/

#clear
#echo "Finished"
