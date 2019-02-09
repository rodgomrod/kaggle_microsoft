from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, IntegerType, LongType)
from pyspark.sql.functions import *
from pyspark import SparkConf
from pyspark import SparkContext
import multiprocessing
from pyspark.ml import Pipeline
import sys

print('Inicio del Script')

# Configuracion de memoria y cores
cores = multiprocessing.cpu_count()
p = 20
particiones = cores * p
conf = SparkConf()
conf.set("spark.sql.shuffle.partitions", particiones)
conf.set("spark.default.parallelism", particiones)
sc = SparkContext(conf=conf)


spark = SparkSession.builder.appName("Microsoft_Kaggle").getOrCreate()


# Label encoding para todas las variables
# exceptuando a las columnas de versiones





final_data = data_final.fillna(imputaciones)

write_path = '../../data/df_cat_prepro_0/'
print('Guardamos el DF en {}'.format(write_path))
# final_data = data.select(['MachineIdentifier'] + cols_transformadas)
final_data.write.csv(write_path, sep=',', mode="overwrite", header=True)
