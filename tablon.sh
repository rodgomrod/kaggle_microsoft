#!/usr/bin/env bash

#echo "Generando DFs con vars categoricas y numericas"
#python3 ETL/Extract/numerical_categoric_dataset.py

#echo "Transformamos variables categoricas"
#python3 ETL/Transform/categorical/global.py

echo "Imputamos valores numericos"
python3 ETL/Transform/numeric/impute_numerical.py

echo "Generamos TRAIN / TEST de las variables nuevas"
python3 ETL/Load/train_test_new_variables.py