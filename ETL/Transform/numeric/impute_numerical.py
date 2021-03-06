from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from pyspark import SparkConf
from pyspark import SparkContext
import multiprocessing


# =============================================================================
# Configuracion de memoria y nº particiones
# =============================================================================
# cores = multiprocessing.cpu_count()
# p = 20
# conf = SparkConf()
# conf.set("spark.driver.cores", cores)
# conf.set("spark.driver.memory", "12g")
# conf.set("spark.sql.shuffle.partitions", p * cores)
# conf.set("spark.default.parallelism", p * cores)
# sc = SparkContext(conf=conf)


# SparkSession
spark = SparkSession.builder.appName('MK_impute_numerical').getOrCreate()




# Read data y persist para mejorar rendimiento
df_num = spark.read.csv('data/df_num/*.csv', header=True, inferSchema=True)
df_num.persist()
df_num.count()

# Algunas medias y medianas para imputar

# Diccionario de imputaciones para todas las columnas que presentan valores nulos
# Las numericas vamos a imputarlas por la mediana
imputaciones = \
    {'RtpStateBitfield': -1, # imputamos por un valor que no se encuentre en la lista
     'DefaultBrowsersIdentifier': 0, # el 0 no existe, lo metemos en esta categoria
     'AVProductStatesIdentifier': 1, # el minimo es 2, metemos los null en el 1
     'AVProductsInstalled': 8, # el maximo es 7
     'AVProductsEnabled': 6, # el maximo es 5
     'CityIdentifier': -1,
     'OrganizationIdentifier': -1,
     'GeoNameIdentifier': 277, # Imputamos por la moda
     'IsProtected': 2, # es 1 o 0, lo metemos en 2 (unknown)
     'SMode': 2, # igual
     'IeVerIdentifier': -1,
     'Firewall': 2,
     'UacLuaenable': -1,
     'Census_OEMNameIdentifier': 0,
     'Census_OEMModelIdentifier': 0,
     'Census_ProcessorManufacturerIdentifier': 0,
     'Census_ProcessorModelIdentifier': 0,
     'Census_OSInstallLanguageIdentifier': 0,
     'Census_IsFlightingInternal': 2,
     'Census_IsFlightsDisabled': 2,
     'Census_FirmwareManufacturerIdentifier': 0,
     'Census_FirmwareVersionIdentifier': 0,
     'Census_IsWIMBootEnabled': 2,
     'Census_IsVirtualDevice': 2,
     'Census_IsAlwaysOnAlwaysConnectedCapable': 2,
     'Wdft_IsGamer': 2,
     'Wdft_RegionIdentifier': 0}

continuous_columns = [
    'Census_ProcessorCoreCount',
    'Census_PrimaryDiskTotalCapacity',
    'Census_SystemVolumeTotalCapacity',
    'Census_TotalPhysicalRAM',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches',
    'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical',
    'Census_InternalBatteryNumberOfCharges',
    'Census_ThresholdOptIn'
]

# Se utiliza la mediana, porque en valores poco informados, la mediana mete menos ruido al modelo que la media.
for c in continuous_columns:
    imputaciones[c] = df_num.approxQuantile(c, [0.5], 0.25)[0]

# Realizamos la imputacion
df_imputado = df_num.fillna(imputaciones)

# Guardamos el DF imputado
write_path = 'data/df_num_imputed_3'
df_imputado.write.csv(write_path, sep=',', mode="overwrite", header=True)