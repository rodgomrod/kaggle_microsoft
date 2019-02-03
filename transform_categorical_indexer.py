from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, IntegerType, LongType)
from pyspark.sql.functions import *
from pyspark import SparkConf
from pyspark import SparkContext
import multiprocessing
from pyspark.ml import Pipeline

print('Inicio del Script')

# Configuracion de memoria y cores
cores = (multiprocessing.cpu_count() - 1)
p = 3
particiones = cores * p
# memoria = 16 # memoria ram instalada
# dm = memoria/2
conf = SparkConf()
conf.set("spark.driver.cores", cores)
conf.set("spark.executor.cores", cores)
conf.set("spark.executor.memory", "10g")
conf.set("spark.driver.memory", "10g")
conf.set("spark.sql.shuffle.partitions", particiones)
conf.set("spark.default.parallelism", particiones)
conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
sc = SparkContext(conf=conf)

# SparkSession
spark = SparkSession.builder.appName("Microsoft_Kaggle").getOrCreate()

# Read data
print('Lectura del DF crudo')
data = spark.read.csv('data/df_cat/*.csv', header=True, inferSchema=True)

# Persistimos el DF para mejorar el rendimiento
data.persist()
print('Numero de casos totales = {}'.format(data.count()))

init_cols = data.columns


# Transformaciones
print('Inicio de las transformaciones:')

## Indexers
    # Label enconding para variables categoricas
columnas_indexer = ['ProductName', 'Census_PrimaryDiskTypeName', 'Census_PowerPlatformRoleName', 'Census_OSArchitecture',
                    'Census_ProcessorClass', 'Census_OSInstallTypeName', 'Census_OSWUAutoUpdateOptionsName',
                    'Census_GenuineStateName', 'Platform', 'Processor', 'OsPlatformSubRelease', 'SkuEdition', 'PuaMode',
                    'Census_DeviceFamily']


print('\tPipeline de Indexers paras las columnas {0}'.format(columnas_indexer))
indexers = [StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid="keep").fit(data) for c in columnas_indexer]
pipeline = Pipeline(stages=indexers)
data0 = pipeline.fit(data).transform(data)

# Imputamos los nulls que hayan quedado
imputaciones = dict()
for c in columnas_indexer:
    imputaciones[c] = -1
data = data0.fillna(imputaciones)

# Persist intermedio
print('Persist intermedio 1')
data.persist()
print('FIRST:\n{}'.format(data.first()))


print('\tCensus_OSVersion')
## Census_OSVersion
    # Al ser una version, se ha hecho split por el punto "."
data = data.withColumn('Census_OSVersion_0', split(data['Census_OSVersion'], '\.')[0].cast(IntegerType()))\
.withColumn('Census_OSVersion_1', split(data['Census_OSVersion'], '\.')[1].cast(IntegerType()))\
.withColumn('Census_OSVersion_2', split(data['Census_OSVersion'], '\.')[2].cast(IntegerType()))\
.withColumn('Census_OSVersion_3', split(data['Census_OSVersion'], '\.')[3].cast(IntegerType()))


print('\tCensus_OSBranch')
## Census_OSBranch
	# frequency
frequency_census = data.groupBy('Census_OSBranch').count().withColumnRenamed('count','Census_OSBranch_freq')
data = data.join(frequency_census,'Census_OSBranch','left')


print('\tEngineVersion')
## EngineVersion
	# Al ser una version, se ha hecho split por el punto "."
    # [0] y [1] es igual para el DF al completo, se ignora
data = data.withColumn('EngineVersion_2', split(data['EngineVersion'], '\.')[2].cast(IntegerType()))\
.withColumn('EngineVersion_3', split(data['EngineVersion'], '\.')[3].cast(IntegerType()))


print('\tAppVersion')
## AppVersion
	# Al ser una version, se ha hecho split por el punto "."
    # [0] es igual para el DF al completo, se ignora
data = data.withColumn('AppVersion_1', split(data['AppVersion'], '\.')[1].cast(IntegerType()))\
.withColumn('AppVersion_2', split(data['AppVersion'], '\.')[2].cast(IntegerType()))\
.withColumn('AppVersion_3', split(data['AppVersion'], '\.')[3].cast(IntegerType()))


print('\tAvSigVersion')
## AvSigVersion
	# Al ser una version, se ha hecho split por el punto "."
    # [3] es igual para el DF al completo, se ignora
data = data.withColumn('AvSigVersion_0', split(data['AvSigVersion'], '\.')[0].cast(IntegerType()))\
.withColumn('AvSigVersion_1', split(data['AvSigVersion'], '\.')[1].cast(IntegerType()))\
.withColumn('AvSigVersion_2', split(data['AvSigVersion'], '\.')[2].cast(IntegerType()))


print('\tOsBuildLab')
## OsBuildLab
	# split por punto "." y transformamos
data1 = data.withColumn('OsBuildLab_0', split(data['OsBuildLab'], '\.')[0].cast(IntegerType()))\
.withColumn('OsBuildLab_1', split(data['OsBuildLab'], '\.')[1].cast(IntegerType()))\
.withColumn('OsBuildLab_2', split(data['OsBuildLab'], '\.')[2])\
.withColumn('OsBuildLab_3', split(data['OsBuildLab'], '\.')[3])\
.withColumn('OsBuildLab_4', split(data['OsBuildLab'], '\.')[4])

data = data1.withColumn('OsBuildLab_4_0', split(data1['OsBuildLab_4'], '-')[0].cast(IntegerType()))\
.withColumn('OsBuildLab_4_1', split(data1['OsBuildLab_4'], '-')[1].cast(IntegerType()))




# Guardamos el DF con las variables categoricas transformadas
final_cols = data.columns
cols_transformadas = list(set(final_cols) - set(init_cols))

write_path = 'data/df_cat_transform_0/indexer_version'
print('Guardamos el DF en {}'.format(write_path))
final_data = data.select(['MachineIdentifier'] + cols_transformadas)
final_data.write.csv(write_path, sep=',', mode="overwrite", header=True)



