from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, IntegerType, LongType)
from pyspark.sql.functions import *
from pyspark import SparkConf
from pyspark import SparkContext
import multiprocessing
from pyspark.ml import Pipeline
import sys

print('Inicio del Script\n')

# =============================================================================
# Configuracion de memoria y nº particiones
# =============================================================================
cores = multiprocessing.cpu_count()
p = 3

spark = SparkSession.builder.appName("Microsoft_Kaggle").getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", p * cores)
spark.conf.set("spark.default.parallelism", p * cores)

data = spark.read.csv('data/df_cat_prepro_0/*.csv', header=True, inferSchema=True)

# Transformaciones
print('Inicio de las transformaciones:\n')

# Label encoding para todas las variables
# exceptuando a las columnas de versiones

cols_le = data.columns
cols_le.remove('MachineIdentifier')
cols_le.remove('Census_OSVersion')
cols_le.remove('EngineVersion')
cols_le.remove('AppVersion')
cols_le.remove('AvSigVersion')
cols_le.remove('OsVer')

data = data.withColumn('Census_InternalBatteryType_informed',
                       when(col('Census_InternalBatteryType').isNotNull(),1).otherwise(0))

print('\Pipeline de Indexers paras las columnas {0}\n'.format(cols_le))
indexers = [StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid="keep").fit(data) for c in cols_le]
pipeline = Pipeline(stages=indexers)
data = pipeline.fit(data).transform(data)
data = data.drop(*cols_le)

print("Persist intermedio 0\n")
data.persist()
print(data.first())

# Transformamos las columnas de versiones "x.y.z.t"
# Conversion a LabelEncoding

# print('Transformacion columnas de versiones\n')
print('\tCensus_OSVersion\n')
data = data.withColumn('Census_OSVersion_0', concat(split(data['Census_OSVersion'], '\.')[0],
                                                split(data['Census_OSVersion'], '\.')[1]))\
    .withColumn('Census_OSVersion_1', concat(split(data['Census_OSVersion'], '\.')[0],
                                     split(data['Census_OSVersion'], '\.')[1],
                                     split(data['Census_OSVersion'], '\.')[2]))

print('\tEngineVersion\n')
data = data.withColumn('EngineVersion_0', split(data['EngineVersion'], '\.')[2])\
.withColumn('EngineVersion_1', concat(split(data['EngineVersion'], '\.')[2],
                                      split(data['EngineVersion'], '\.')[3]))

print('\tAppVersion\n')
data = data.withColumn('AppVersion_0', concat(split(data['AppVersion'], '\.')[1],
                                                split(data['AppVersion'], '\.')[2]))\
    .withColumn('AppVersion_1', concat(split(data['AppVersion'], '\.')[1],
                                     split(data['AppVersion'], '\.')[2],
                                     split(data['AppVersion'], '\.')[3]))

print('\tAvSigVersion\n')
data = data.withColumn('AvSigVersion_0', concat(split(data['AvSigVersion'], '\.')[0],
                                                split(data['AvSigVersion'], '\.')[1]))\
    .withColumn('AvSigVersion_1', concat(split(data['AvSigVersion'], '\.')[0],
                                     split(data['AvSigVersion'], '\.')[1],
                                     split(data['AvSigVersion'], '\.')[2]))

print('\tOsVer\n')
data = data.withColumn('OsVer_0', concat(split(data['OsVer'], '\.')[0],
                                                split(data['OsVer'], '\.')[1]))\
    .withColumn('OsVer_1', concat(split(data['OsVer'], '\.')[0],
                                     split(data['OsVer'], '\.')[1],
                                     split(data['OsVer'], '\.')[2]))


drop_cols_2 = ['Census_OSVersion', 'EngineVersion', 'AppVersion', 'AvSigVersion', 'OsVer', 'Census_OSVersion_0', 'Census_OSVersion_1',
               'EngineVersion_0', 'EngineVersion_1', 'AppVersion_0', 'AppVersion_1', 'AvSigVersion_0', 'AvSigVersion_1', 'OsVer_0', 'OsVer_1']

indexers = [StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid="keep").fit(data) for c in drop_cols_2]
pipeline = Pipeline(stages=indexers)
data = pipeline.fit(data).transform(data)

# El * es para que itere sobre la lista, así nos ahorramos hacer un for. Si fuera un diccionario, se pone **
data = data.drop(*drop_cols_2)


write_path = 'data/df_cat_pro_3'
print('Guardamos el DF en {}\n'.format(write_path))
data.write.csv(write_path, sep=',', mode="overwrite", header=True)

