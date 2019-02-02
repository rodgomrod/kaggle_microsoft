from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark import SparkConf
from pyspark import SparkContext
import multiprocessing

# Configuracion de memoria y cores
cores = multiprocessing.cpu_count()
p = 15
conf = SparkConf()
conf.set("spark.driver.cores", cores)
conf.set("spark.driver.memory", "10g")
conf.set("spark.sql.shuffle.partitions", p * cores)
conf.set("spark.default.parallelism", p * cores)
sc = SparkContext(conf=conf)

# SparkSession
spark = SparkSession.builder.appName('MK_impute_numerical').getOrCreate()

# Read data y persist para mejorar rendimiento
df_num = spark.read.csv('data/df_num/*.csv', header=True, inferSchema=True)
df_num.persist()
df_num.count()

# Algunas medias y medianas para imputar
mediana_GeoNameIdentifier = df_num.approxQuantile("GeoNameIdentifier", [0.5], 0.25)[0]
media_RtpStateBitfield = df_num.agg(avg("RtpStateBitfield")).collect()[0][0]

# Diccionario de imputaciones para todas las columnas que presentan valores nulos
imputaciones = \
{'RtpStateBitfield': media_RtpStateBitfield, # tiene STDDEV = 1 por lo que la media puede resultar
 'DefaultBrowsersIdentifier': 0, # el 0 no existe, lo metemos en esta categoria
 'AVProductStatesIdentifier': 1, # el minimo es 2, metemos los null en el 1
 'AVProductsInstalled': 8, # el maximo es 8
 'AVProductsEnabled': 6, # el maximo es 5
 'CityIdentifier': -99,
 'OrganizationIdentifier': -99,
 'GeoNameIdentifier': mediana_GeoNameIdentifier, # Al ser pocos casos, le meteremos la mediana
 'IsProtected': 2, # es 1 o 0, lo metemos en 2 (unknown)
 'SMode': 2, # igual
 'IeVerIdentifier': -99,
 'Firewall': 2,
 'UacLuaenable': -99,
 'Census_OEMNameIdentifier': 0,
 'Census_OEMModelIdentifier': 0,
 'Census_ProcessorCoreCount': -99,
 'Census_ProcessorManufacturerIdentifier': 0,
 'Census_ProcessorModelIdentifier': 0,
 'Census_PrimaryDiskTotalCapacity': -99,
 'Census_SystemVolumeTotalCapacity': -99,
 'Census_TotalPhysicalRAM': -99,
 'Census_InternalPrimaryDiagonalDisplaySizeInInches': -99,
 'Census_InternalPrimaryDisplayResolutionHorizontal': -99,
 'Census_InternalPrimaryDisplayResolutionVertical': -99,
 'Census_InternalBatteryNumberOfCharges': -99,
 'Census_OSInstallLanguageIdentifier': 0,
 'Census_IsFlightingInternal': 2,
 'Census_IsFlightsDisabled': 2,
 'Census_ThresholdOptIn': 2,
 'Census_FirmwareManufacturerIdentifier': 0,
 'Census_FirmwareVersionIdentifier': 0,
 'Census_IsWIMBootEnabled': 2,
 'Census_IsVirtualDevice': 2,
 'Census_IsAlwaysOnAlwaysConnectedCapable': 2,
 'Wdft_IsGamer': 2,
 'Wdft_RegionIdentifier': 0}

# Realizamos la imputacion
df_imputado = df_num.fillna(imputaciones)

# Guardamos el DF imputado
write_path = 'data/df_num_imputed_0'
df_imputado.write.csv(write_path, sep=',', mode="overwrite", header=True)