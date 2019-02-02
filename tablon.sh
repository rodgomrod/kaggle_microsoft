#!/usr/bin/env bash

echo "Generando DFs con vars categoricas y numericas"
python3 generador_num_cat.py

echo "Transformamos variables categoricas"
python3 transform_categorical.py

echo "Imputamos valores numericos"
python3 impute_numerical.py

echo "Generamos TRAIN / TEST final"
python3 genera_train_test_final.py