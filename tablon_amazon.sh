#!/usr/bin/env bash
clear
echo "1/6"
echo -e "\e[37;1m\e[5mGenerando DFs con vars categoricas y numericas\e[0m"
python3 ETL/Extract/numerical_categoric_dataset.py
echo "-------------------"



clear
echo "2/6"
echo -e "\e[37;1m\e[5mPreprocesamos variables categoricas\e[0m"
python3 ETL/Preprocessing/categorical.py
echo "-------------------"

# rm -rf data/df_cat/


clear
echo "3/6"
echo -e "\e[37;1m\e[5mProcesamos variables categoricas\e[0m"
python3 ETL/Transform/categorical/global.py
echo "-------------------"

# rm -rf data/df_cat_prepro_0/

clear
echo "4/6"
echo -e "\e[37;1m\e[5mImputamos valores numericos\e[0m"
python3 ETL/Transform/numeric/impute_numerical.py
echo "-------------------"

# rm -rf data/df_num/


clear
echo "5/6"
echo -e "\e[37;1m\e[5mProcesamos variables de fechas\e[0m"
python3 ETL/Transform/dates/dates.py
echo ""


clear
echo "6/6"
echo -e "\e[37;1m\e[5mGeneramos TRAIN / TEST de las variables nuevas\e[0m"
python3 ETL/Load/train_test_new_variables.py
echo "-------------------"

# rm -rf  data/df_cat_pro_0/
# rm -rf  data/df_num_imputed_0/

clear
echo "Finished"