from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, IntegerType, LongType)
import multiprocessing

# conf = SparkConf()
# cores = multiprocessing.cpu_count()
# conf = SparkConf()
# conf.set("spark.sql.shuffle.partitions", int(800))
# conf.set("spark.default.parallelism", int(800))
# # conf.set("spark.driver.cores", int(8))
# # conf.set("spark.executor.cores ", int(8))
# conf.set("spark.driver.memory", '15g')
# # conf.set("spark.executor.memory", '15g')
#
# sc = SparkContext(conf=conf)

spark = SparkSession.builder.appName("Microsoft_Kaggle").getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", 160)
spark.conf.set("spark.default.parallelism", 160)
spark.conf.set("spark.shuffle.consolidateFiles", "false")

df_num = spark.read.csv("data/df_cat_prepro_0/*.csv",inferSchema=True,header=True)\
    .select('MachineIdentifier','Platform', 'OsBuildLab', 'AvSigVersion', 'Census_OSVersion')
# df_num.persist()
# df_num.count()
print('DF leido')

df_fechas_av = spark.read.csv("Notebooks/new_variables/fechas_av.csv",inferSchema=True,header=True)
df_fechas_os = spark.read.csv("Notebooks/new_variables/fechas_os.csv",inferSchema=True,header=True)

df_fechas_os = df_fechas_os.withColumn('DateOSVersion', to_date(col('DateCensus_OSVersion')))
df_fechas_av = df_fechas_av.withColumn('DateAvSigVersion', to_date(col('DateAvSigVersion')))

# df_fechas_os.persist()
# df_fechas_os.count()
#
# df_fechas_av.persist()
# df_fechas_av.count()

print('Fechas leidas')

df_date_osbuild = df_num.withColumn('OsBuildLab_4', split(df_num['OsBuildLab'], '\.')[4].cast(StringType()))
df_date_osbuild = df_date_osbuild.withColumn('OsBuildLab_date', split(df_date_osbuild['OsBuildLab_4'], '-')[0].cast(StringType()))

df_date_osbuild = df_date_osbuild.withColumn('DateOsBuildLab', to_date(col('OsBuildLab_date'), format='yyMMdd'))

df_dates = df_date_osbuild.join(df_fechas_av, ['AvSigVersion'], 'left').select('MachineIdentifier',
                                                                               'Platform',
                                                                               'DateOsBuildLab',
                                                                               'DateAvSigVersion',
                                                                               'Census_OSVersion')

# df_dates.persist()
# df_dates.count()

df_dates = df_dates.join(df_fechas_os, ['Census_OSVersion'], 'left').select('MachineIdentifier',
                                                                               'Platform',
                                                                               'DateOsBuildLab',
                                                                               'DateAvSigVersion',
                                                                           'DateOSVersion').repartition(160, ['MachineIdentifier'])

df_dates.persist()
print(df_dates.count())

print("join entre dates y DF")

w1 = Window.partitionBy('Platform').orderBy('DateOsBuildLab')
w2 = Window.partitionBy('Platform').orderBy('DateAvSigVersion')
w5 = Window.partitionBy('Platform').orderBy('DateOSVersion')
w3 = Window.partitionBy('MachineIdentifier').orderBy('DateOsBuildLab')
w4 = Window.partitionBy('MachineIdentifier').orderBy('DateAvSigVersion')

data_windows = df_dates.withColumn('DateOsBuildLab_lag', lag('DateOsBuildLab').over(w1))
print("window 1")
data_windows = data_windows.withColumn('OsBuildLab_diff', datediff(col('DateOsBuildLab'), col('DateOsBuildLab_lag')))
# data_windows.persist()
# print(data_windows.count())

data_windows1 = data_windows.withColumn('DateAvSigVersion_lag', lag('DateAvSigVersion').over(w2))
print("window 2")
data_windows1 = data_windows1.withColumn('AvSigVersion_diff', datediff(col('DateAvSigVersion'), col('DateAvSigVersion_lag')))
# data_windows.persist()
# print(data_windows.count())

data_windows2 = data_windows1.withColumn('DateOSVersion_lag', lag('DateOSVersion').over(w5))
print("window 3")
data_windows2 = data_windows2.withColumn('OSVersion_diff', datediff(col('DateOSVersion'), col('DateOSVersion_lag')))
# data_windows.persist()
# print(data_windows.count())

data_windows3 = data_windows2.withColumn('DateOsBuildLab_fulllag', lag('DateOsBuildLab').over(w3))
print("window 4")
data_windows3 = data_windows3.withColumn('OSBuild_fulldiff', datediff(col('DateOsBuildLab'), col('DateOsBuildLab_fulllag')))
# data_windows.persist()
# print(data_windows.count())

data_windows4 = data_windows3.withColumn('DateAvSigVersion_fulllag', lag('DateAvSigVersion').over(w4))
print("window 5")
data_windows4 = data_windows4.withColumn('AvSigVersion_fulldiff', datediff(col('DateAvSigVersion'), col('DateAvSigVersion_fulllag')))
# data_windows.persist()
# print(data_windows.count())


df_max_date = data_windows4.groupBy('Platform').agg(max('DateOsBuildLab'),
                                                    max('DateAvSigVersion'),
                                                    max('DateOSVersion'),
                                                    max('OsBuildLab_diff'),
                                                    max('AvSigVersion_diff'),
                                                    max('OSVersion_diff'),
                                                    max('OSBuild_fulldiff'),
                                                    max('AvSigVersion_fulldiff'))

df_date_max_date = data_windows4.join(df_max_date, ['Platform'], 'left').repartition(160, ['MachineIdentifier'])
df_date_max_date.persist()
print(df_date_max_date.count())

print("ultimo join")

df_date_max_date_final = df_date_max_date.withColumn('OsBuildLab_difftotal', datediff(col('max(DateOsBuildLab)'), col('DateOsBuildLab')))\
.withColumn('DateAvSigVersion_difftotal', datediff(col('max(DateAvSigVersion)'), col('DateAvSigVersion')))\
.withColumn('DateOSVersion_difftotal', datediff(col('max(DateOSVersion)'), col('DateOSVersion')))\
.withColumn('DateAvSigVersion_fulldifftotal', datediff(col('max(DateAvSigVersion)'), col('DateAvSigVersion_fulllag')))\
.withColumn('OsBuildLab_fulldifftotal', datediff(col('max(DateOSVersion)'), col('DateOsBuildLab_fulllag')))\
.withColumn('DateAvSigVersion_ratio', col('AvSigVersion_diff')/col('max(AvSigVersion_diff)'))\
.withColumn('OsBuildLab_ratio', col('OsBuildLab_diff')/col('max(OsBuildLab_diff)'))\
.withColumn('OSVersion_ratio', col('OSVersion_diff')/col('max(OSVersion_diff)'))\
.withColumn('DateAvSigVersion_fullratio', col('AvSigVersion_fulldiff')/col('max(AvSigVersion_diff)'))\
.withColumn('OsBuildLab_fullratio', col('OSBuild_fulldiff')/col('max(OSBuild_fulldiff)'))

final_dates = df_date_max_date_final.withColumn('OsBuildLab_dayOfWeek', date_format('DateOsBuildLab', 'u'))\
.withColumn('AvSigVersion_dayOfWeek', date_format('DateAvSigVersion', 'u'))


drop_list = ['Platform', 'DateOsBuildLab', 'DateAvSigVersion', 'DateOsBuildLab_lag', 'DateOSVersion',
             'DateOSVersion_lag', 'max(DateOSVersion)', 'max(OSVersion_diff)',
             'DateAvSigVersion_lag', 'max(DateOsBuildLab)', 'max(DateAvSigVersion)',
             'DateOsBuildLab_fulllag', 'DateAvSigVersion_fulllag', 'max(OsBuildLab_diff)',
             'max(AvSigVersion_diff)', 'max(OSBuild_fulldiff)', 'max(AvSigVersion_fulldiff)']

for c in drop_list:
    final_dates = final_dates.drop(c)

final_dates_imputed = final_dates.fillna(0).repartition(160, ['MachineIdentifier'])

write_path = 'data/df_dates_2'
print('Guardamos el DF en {}'.format(write_path))
final_dates_imputed.write.csv(write_path, sep=',', mode="overwrite", header=True)
