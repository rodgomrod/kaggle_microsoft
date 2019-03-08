from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, IntegerType, LongType)
from pyspark.sql.functions import *


print('Inicio del Script\n')

N1 = ['Census_OSEdition', 'SmartScreen', 'Census_OSBranch', 'Census_OSSkuName']

N2 = ['Census_ChassisTypeName', 'Census_FlightRing', 'Census_MDC2FormFactor', 'ProductName']

N3 = ['Census_PrimaryDiskTypeName', 'Census_PowerPlatformRoleName', 'Census_ProcessorClass']

N4 = ['Processor', 'Census_OSInstallTypeName', 'OsVer', 'Census_GenuineStateName', 'PuaMode']

N5 = ['Census_OSVersion', 'EngineVersion', 'AppVersion', 'AvSigVersion']

write_path = 'data/df_groupby_cat_0'

spark = SparkSession.builder.appName("Microsoft_Kaggle").getOrCreate()

data = spark.read.csv('data/df_cat_prepro_0/*.csv', header=True, inferSchema=True).select(N1+N2+N3+N4+N5+['MachineIdentifier'])

print('Guardamos el DF en {}\n'.format(write_path))
data.join(data.groupBy(N1).agg(count('*').alias('count1')), N1, 'left')\
    .join(data.groupBy(N2).agg(count('*').alias('count2')), N2, 'left')\
    .join(data.groupBy(N3).agg(count('*').alias('count3')), N3, 'left')\
    .join(data.groupBy(N4).agg(count('*').alias('count4')), N4, 'left')\
    .join(data.groupBy(N5).agg(count('*').alias('count5')), N5, 'left')\
    .select('MachineIdentifier', 'count1', 'count2', 'count3', 'count4', 'count5')\
    .write.csv(write_path, sep=',', mode="overwrite", header=True)



