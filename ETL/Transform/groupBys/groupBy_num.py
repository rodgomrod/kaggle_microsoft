from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, IntegerType, LongType)
from pyspark.sql.functions import *


print('Inicio del Script\n')

N6 = ['Wdft_IsGamer', 'IsBeta', 'IsSxsPassiveMode', 'IsProtected', 'AVProductsEnabled',
'AVProductsInstalled', 'HasTpm']

N7 = ['DefaultBrowsersIdentifier', 'AVProductStatesIdentifier', 'CountryIdentifier',
'CityIdentifier', 'OrganizationIdentifier', 'GeoNameIdentifier']

N8 = ['Census_OEMNameIdentifier', 'Census_OEMModelIdentifier', 'Census_ProcessorManufacturerIdentifier',
'Census_ProcessorModelIdentifier', 'Census_OSInstallLanguageIdentifier', 'Census_FirmwareVersionIdentifier',
'Census_FirmwareManufacturerIdentifier']

write_path = 'data/df_groupby_num_0'

spark = SparkSession.builder.appName("Microsoft_Kaggle").getOrCreate()

data = spark.read.csv('data/df_num_imputed_3/*.csv', header=True, inferSchema=True).select(N6+N7+N8+['MachineIdentifier'])

print('Guardamos el DF en {}\n'.format(write_path))
data.join(data.groupBy(N6).agg(count('*').alias('count6')), N6, 'left')\
    .join(data.groupBy(N7).agg(count('*').alias('count7')), N7, 'left')\
    .join(data.groupBy(N8).agg(count('*').alias('count8')), N8, 'left')\
    .select('MachineIdentifier', 'count6', 'count7', 'count8')\
    .write.csv(write_path, sep=',', mode="overwrite", header=True)