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
# # conf.set("spark.driver.cores", cores)
# conf.set("spark.driver.memory", "12g")
# conf.set("spark.sql.shuffle.partitions", p * cores)
# conf.set("spark.default.parallelism", p * cores)
# sc = SparkContext(conf=conf)

# SparkSession
spark = SparkSession.builder.appName('MK_genera_train_test').getOrCreate()

df_num = spark.read.csv('data/df_num_imputed_3/*.csv', header=True, inferSchema=True)
df_num.persist()
df_num.count()

# cat_cols = ['MachineIdentifier', 'Census_InternalBatteryType_informed', 'ProductName_index', 'Platform_index',
#             'Processor_index', 'OsPlatformSubRelease_index', 'SkuEdition_index', 'PuaMode_index', 'SmartScreen_index',
#             'Census_MDC2FormFactor_index', 'Census_DeviceFamily_index', 'Census_ProcessorClass_index',
#             'Census_PrimaryDiskTypeName_index', 'Census_ChassisTypeName_index', 'Census_PowerPlatformRoleName_index',
#             'Census_InternalBatteryType_index', 'Census_OSArchitecture_index', 'Census_OSBranch_index',
#             'Census_OSEdition_index', 'Census_OSSkuName_index', 'Census_OSInstallTypeName_index',
#             'Census_OSWUAutoUpdateOptionsName_index', 'Census_GenuineStateName_index', 'Census_ActivationChannel_index',
#             'Census_FlightRing_index']

df_cat = spark.read.csv('data/df_cat_pro_3/*.csv', header=True, inferSchema=True)
# df_cat.persist()
# df_cat.count()

df_dates = spark.read.csv('data/df_dates_2/*.csv', header=True, inferSchema=True)
# df_dates.persist()
# df_dates.count()

df_kmeans = spark.read.csv('data/df_kmeans_2/*.csv', header=True, inferSchema=True)
# df_kmeans.persist()
# df_kmeans.count()

df_avsigver = spark.read.csv('data/df_avsig_version/*.csv', header=True, inferSchema=True)
# df_avsigver.persist()
# df_avsigver.count()


full_df = df_num.join(df_cat, ['MachineIdentifier'])\
                .join(df_dates, ['MachineIdentifier'])\
                .join(df_kmeans, ['MachineIdentifier'])\
                .join(df_avsigver, ['MachineIdentifier'])

# full_df2 = full_df_1.join(df_dates, ['MachineIdentifier'])
#
# full_df3 = full_df2.join(df_kmeans, ['MachineIdentifier'])
#
# full_df = full_df3.join(df_avsigver, ['MachineIdentifier'])


# full_df.persist()
# full_df.count()

train = full_df.filter(col('HasDetections').isNotNull())
train = train.fillna(-1)
test = full_df.filter(col('HasDetections').isNull())
test = test.fillna(-1)

write_path_train = 'data/train_final_3'
write_path_test = 'data/test_final_3'
train.write.csv(write_path_train, sep=',', mode="overwrite", header=True)
test.write.csv(write_path_test, sep=',', mode="overwrite", header=True)
