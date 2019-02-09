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

#####################################################################################

# Funciones para las transformaciones de ciertos valores en columnas

def transformaciones_ChassisTypeName(x):
    try:
        to_int = int(x)
        return 'Numerico'
    except:
        if x == 'Unknown' or x == 'Other':
            return 'UNKNOWN'
        else:
            return x

udf_ChassisTypeName = udf(lambda z: transformaciones_ChassisTypeName(z), StringType())


def transformaciones_OSEdition(x):
    if x == '#':
        return None
    elif x == '00426-OEM-8992662-00006':
        return 'Ultimate'
    elif x == 'HomePremium' or x == 'HomeBasic':
        return 'Home'
    elif x == 'Window 10 Enterprise' or x == 'Enterprise 2015 LTSB' or x == 'EnterpriseN' \
            or x == 'EnterpriseG':
        return 'Enterprise'
    elif x == 'ServerDatacenterACor' or x == 'ServerDatacenterEval':
        return 'ServerDatacenter'
    elif x == 'ProfessionalSingleLanguage' or x == 'PRO' or x == 'Pro' or x == 'professional' \
            or x == 'ProfessionalCountrySpecific':
        return 'Professional'
    elif x == 'ProfessionalEducationN' or x == 'EducationN' or x == 'ProfessionalEducation':
        return 'Education'
    elif x == 'CloudN':
        return 'Cloud'
    elif x == 'ProfessionalWorkstationN':
        return 'ProfessionalWorkstation'
    else:
        return x

udf_OSEdition = udf(lambda z: transformaciones_OSEdition(z), StringType())


def transformaciones_SmartScreen(x):
    if x == '&#x01;' or x == '0' or x == '00000000' or x == '&#x02;' or x == '&#x03;':
        return None
    elif x == 'Block':
        return 'BLOCK'
    elif x == 'requireadmin' or x == 'RequireAdmin' or x == 'requireAdmin':
        return 'RequiredAdmin'
    elif x == 'Promt' or x == 'prompt' or x == 'Promprt':
        return 'Prompt'
    elif x == 'of' or x == 'Off' or x == 'off':
        return 'OFF'
    elif x == 'on' or x == 'On' or x == 'Enabled':
        return 'ON'
    elif x == 'warn':
        return 'Warn'
    else:
        return x

udf_SmartScreen = udf(lambda z: transformaciones_SmartScreen(z), StringType())

#####################################################################################

print('Lectura del DF crudo')
data = spark.read.csv('../../data/df_cat/*.csv', header=True, inferSchema=True)\
    .withColumn('Census_PrimaryDiskTypeName',
                    when((col('Census_PrimaryDiskTypeName').isNull()) |\
                         (col('Census_PrimaryDiskTypeName') == 'Unspecified'), 'UNKNOWN')\
                    .otherwise(col('Census_PrimaryDiskTypeName')))\
    .withColumn('Census_PowerPlatformRoleName',
                    when((col('Census_PowerPlatformRoleName').isNull()) |\
                         (col('Census_PowerPlatformRoleName') == 'Unspecified'), 'UNKNOWN')\
                   .otherwise(col('Census_PowerPlatformRoleName')))\
    .withColumn('Census_ProcessorClass',
                    when(col('Census_ProcessorClass').isNull(), 'UNKNOWN')\
                   .otherwise(col('Census_ProcessorClass')))\
    .withColumn('Census_GenuineStateName',
                    when(col('Census_GenuineStateName').isNull(), 'UNKNOWN')\
                   .otherwise(col('Census_GenuineStateName')))\
    .withColumn('PuaMode',
                    when(col('PuaMode').isNull(), 'UNKNOWN')\
                   .otherwise(col('PuaMode')))

print('Persist intermedio 0')
data.persist()
print(data.first())

data_final = data\
    .withColumn('Census_ChassisTypeName', udf_ChassisTypeName('Census_ChassisTypeName'))\
    .withColumn('Census_OSEdition', udf_OSEdition('Census_OSEdition'))\
    .withColumn('SmartScreen', udf_SmartScreen('SmartScreen'))

imputaciones = {
    'SmartScreen': 'Unknown',
    'Census_InternalBatteryType': 'Unknown',
    'Census_ChassisTypeName': 'UNKNOWN',
    'OsBuildLab': '0.0.0.0.0-0',
    'Census_OSEdition': 'Unknown'
}

final_data = data_final.fillna(imputaciones)

write_path = '../../data/df_cat_prepro_0/'
print('Guardamos el DF en {}'.format(write_path))
# final_data = data.select(['MachineIdentifier'] + cols_transformadas)
final_data.write.csv(write_path, sep=',', mode="overwrite", header=True)

