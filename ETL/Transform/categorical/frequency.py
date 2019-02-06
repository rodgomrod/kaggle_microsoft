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
p = 30
particiones = cores * p
# memoria = 16 # memoria ram instalada
# dm = memoria/2
conf = SparkConf()
conf.set("spark.driver.cores", cores)
conf.set("spark.executor.cores", cores)
# conf.set("spark.executor.memory", "10g")
# conf.set("spark.driver.memory", "3g")
conf.set("spark.sql.shuffle.partitions", particiones)
conf.set("spark.default.parallelism", particiones)
sc = SparkContext(conf=conf)

# c = sys.argv[1]
# TODO: estudiar los nulls y los unknown

# SparkSession
spark = SparkSession.builder.appName("Microsoft_Kaggle").getOrCreate()

# Read data
print('Lectura del DF crudo')
data = spark.read.csv('../../../data/df_cat/*.csv', header=True, inferSchema=True)\
.select('MachineIdentifier', 'Census_ChassisTypeName', 'Census_InternalBatteryType', 'Census_OSBranch', 'Census_OSEdition', 'Census_OSSkuName', 'Census_FlightRing', 'OsVer', 'SmartScreen', 'Census_MDC2FormFactor')

# Persistimos el DF para mejorar el rendimiento
data.persist()
print('Numero de casos totales = {}'.format(data.count()))

init_cols = data.columns


# Transformaciones
print('Inicio de las transformaciones:')

print('\tCensus_ChassisTypeName')
## Census_ChassisTypeName
	# Frecuencia
frequency_census = data.groupBy('Census_ChassisTypeName').count().withColumnRenamed('count','Census_ChassisTypeName_freq')
data = data.join(frequency_census,'Census_ChassisTypeName','left').drop('Census_ChassisTypeName')

data.persist()
print(data.first())


print('\tCensus_InternalBatteryType bool')
## Census_InternalBatteryType
	# Frecuencia
frequency_census = data.groupBy('Census_InternalBatteryType').count().withColumnRenamed('count','Census_InternalBatteryType_freq')
data = data.join(frequency_census,'Census_InternalBatteryType','left')

	#Booleana. QUITAR EN EL FUTURO EL NULL
data = data.withColumn('Census_InternalBatteryType_informed', when(col('Census_InternalBatteryType').isNotNull(),1).otherwise(0)).drop('Census_InternalBatteryType')

data.persist()
print(data.first())

print('\tCensus_OSBranch')
## Census_OSBranch
	# frequency
frequency_census = data.groupBy('Census_OSBranch').count().withColumnRenamed('count','Census_OSBranch_freq')
data = data.join(frequency_census,'Census_OSBranch','left').drop('Census_OSBranch')

data.persist()
print(data.first())

print('\tCensus_OSEdition')
## Census_OSEdition
	# Frecuencia
df_cat_freq_Census_OSEdition = data.groupBy('Census_OSEdition').count().withColumnRenamed('count', 'Census_OSEdition_freq')
data = data.join(df_cat_freq_Census_OSEdition, ['Census_OSEdition'], 'left').drop('Census_OSEdition')

data.persist()
print(data.first())

print('\tCensus_OSSkuName')
## Census_OSSkuName
	# Frecuencia
frequency_Census_OSSkuName = data.groupBy('Census_OSSkuName').count().withColumnRenamed('count','Census_OSSkuName_freq')
data = data.join(frequency_Census_OSSkuName,'Census_OSSkuName','left').drop('Census_OSSkuName')

data.persist()
print(data.first())

print('\tCensus_FlightRing')
## Census_FlightRing
	# Frecuencia
frequency_Census_Census_FlightRing = data.groupBy('Census_FlightRing').count().withColumnRenamed('count','Census_FlightRing_freq')
data = data.join(frequency_Census_Census_FlightRing,'Census_FlightRing','left').drop('Census_FlightRing')

data.persist()
print(data.first())

print('\tOsVer')
## OsVer
	# Frecuencia
df_cat_freq_osver = data.groupBy('OsVer').count().withColumnRenamed('count', 'OsVer_freq')
data = data.join(df_cat_freq_osver, ['OsVer'], 'left').drop('OsVer')

data.persist()
print(data.first())

print('\tSmartScreen')
## SmartScreen
	# Frecuencia
df_cat_freq_SmartScreen = data.groupBy('SmartScreen').count().withColumnRenamed('count', 'SmartScreen_freq')
data = data.join(df_cat_freq_SmartScreen, ['SmartScreen'], 'left').drop('SmartScreen')

data.persist()
print(data.first())

print('\tCensus_MDC2FormFactor')
## Census_MDC2FormFactor
	# Frecuencia
df_cat_freq_osver = data.groupBy('Census_MDC2FormFactor').count().withColumnRenamed('count', 'Census_MDC2FormFactor_freq')
data = data.join(df_cat_freq_osver, ['Census_MDC2FormFactor'], 'left').drop('Census_MDC2FormFactor')

data.persist()
print(data.first())

# Guardamos el DF con las variables categoricas transformadas
final_cols = data.columns
cols_transformadas = list(set(final_cols) - set(init_cols))

print('Columnas transformadas: {}'.format(cols_transformadas))

imputaciones = dict()
for c in final_cols:
    imputaciones[c] = -1
final_data = data.fillna(imputaciones)

write_path = '../../../data/df_cat_transform_0/freq'
print('Guardamos el DF en {}'.format(write_path))
# final_data = data.select(['MachineIdentifier'] + cols_transformadas)
final_data.write.csv(write_path, sep=',', mode="overwrite", header=True)



