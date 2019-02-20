from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, IntegerType, LongType)
import multiprocessing


conf = SparkConf()
cores = multiprocessing.cpu_count()
conf = SparkConf()
conf.set("spark.sql.shuffle.partitions", int(800))
conf.set("spark.default.parallelism", int(800))
# conf.set("spark.driver.cores", int(8))
# conf.set("spark.executor.cores ", int(8))
conf.set("spark.driver.memory", '15g')
# conf.set("spark.executor.memory", '15g')

sc = SparkContext(conf=conf)

spark = SparkSession.builder.appName("Microsoft_Kaggle").getOrCreate()

df_num = spark.read.csv("../../../data/df_cat_prepro_0/*.csv",inferSchema=True,header=True)\
    .select('MachineIdentifier','Platform', 'OsBuildLab', 'AvSigVersion')
df_num.persist()
df_num.count()
print('DF leido')

df_fechas_av = spark.read.csv("../../../Notebooks/fechas_av.csv",inferSchema=True,header=True)
# df_fechas_os = spark.read.csv("../../../Notebooks/fechas_os.csv",inferSchema=True,header=True)

# df_fechas_os = df_fechas_os.withColumn('DateCensus_OSVersion', to_date(col('DateCensus_OSVersion')))
df_fechas_av = df_fechas_av.withColumn('DateAvSigVersion', to_date(col('DateAvSigVersion')))

# df_fechas_os = df_fechas_os.withColumn('DateCensus_OSVersion', to_date(col('DateCensus_OSVersion')))
df_fechas_av = df_fechas_av.withColumn('DateAvSigVersion', to_date(col('DateAvSigVersion')))

print('Fechas AV leido')

df_date_osbuild = df_num.withColumn('OsBuildLab_4', split(df_num['OsBuildLab'], '\.')[4].cast(StringType()))
df_date_osbuild = df_date_osbuild.withColumn('OsBuildLab_date', split(df_date_osbuild['OsBuildLab_4'], '-')[0].cast(StringType()))

df_date_osbuild = df_date_osbuild.withColumn('OsBuildLab_date', to_date(col('OsBuildLab_date'), format='yyMMdd'))

df_dates = df_date_osbuild.join(df_fechas_av, ['AvSigVersion'], 'left').select('MachineIdentifier',
                                                                               'Platform',
                                                                               'OsBuildLab_date',
                                                                               'DateAvSigVersion')

print("join entre AV dates y DF")

w1 = Window.partitionBy('Platform').orderBy('OsBuildLab_date')
w2 = Window.partitionBy('Platform').orderBy('DateAvSigVersion')
w3 = Window.partitionBy().orderBy('OsBuildLab_date')
w4 = Window.partitionBy().orderBy('DateAvSigVersion')

data_windows = df_dates.withColumn('OsBuildLab_date_lag', lag('OsBuildLab_date').over(w1))
# data_windows.persist()
# data_windows.count()
print("window 1")

# df_num.unpersist()
# spark.catalog.clearCache()

date_diff = data_windows.withColumn('OSBuild_diff', datediff(col('OsBuildLab_date'), col('OsBuildLab_date_lag')))

date_diff_2 = date_diff.withColumn('DateAvSigVersion_lag', lag('DateAvSigVersion').over(w2))
# date_diff_2.persist()
# date_diff_2.count()
print("window 2")

date_diff_3 = date_diff_2.withColumn('AvSigVersion_diff', datediff(col('DateAvSigVersion'), col('DateAvSigVersion_lag')))

data_windows = date_diff_3.withColumn('OsBuildLab_date_fulllag', lag('OsBuildLab_date').over(w3))
# data_windows.persist()
# data_windows.count()
print("window 3")

data_windows = data_windows.withColumn('OSBuild_fulldiff', datediff(col('OsBuildLab_date'), col('OsBuildLab_date_fulllag')))

data_windows = data_windows.withColumn('DateAvSigVersion_fulllag', lag('DateAvSigVersion').over(w4))
data_windows.persist()
data_windows.count()
print("window 4")

data_windows = data_windows.withColumn('AvSigVersion_fulldiff', datediff(col('DateAvSigVersion'), col('DateAvSigVersion_fulllag')))


df_max_date = data_windows.groupBy('Platform').agg(max('OsBuildLab_date'),
                                                   max('DateAvSigVersion'),
                                                   max('OSBuild_diff'),
                                                   max('AvSigVersion_diff'),
                                                   max('OSBuild_fulldiff'),
                                                   max('AvSigVersion_fulldiff'))

df_date_max_date = data_windows.join(df_max_date, ['Platform'], 'left')

print("ultimo join")

df_date_max_date = df_date_max_date.withColumn('OsBuildLab_difftotal', datediff(col('max(OsBuildLab_date)'), col('OsBuildLab_date')))\
.withColumn('DateAvSigVersion_difftotal', datediff(col('max(DateAvSigVersion)'), col('DateAvSigVersion')))\
.withColumn('DateAvSigVersion_fulldifftotal', datediff(col('max(DateAvSigVersion)'), col('DateAvSigVersion_fulllag')))\
.withColumn('OsBuildLab_fulldifftotal', datediff(col('max(OsBuildLab_date)'), col('OsBuildLab_date_fulllag')))\
.withColumn('DateAvSigVersion_ratio', round(col('AvSigVersion_diff')/col('max(AvSigVersion_diff)')*100, 2))\
.withColumn('OsBuildLab_ratio', round(col('OSBuild_diff')/col('max(OSBuild_diff)')*100, 2))\
.withColumn('DateAvSigVersion_fullratio', round(col('AvSigVersion_fulldiff')/col('max(AvSigVersion_diff)')*100, 2))\
.withColumn('OsBuildLab_fullratio', round(col('OSBuild_fulldiff')/col('max(OSBuild_diff)')*100, 2))

final_dates = df_date_max_date.withColumn('OsBuildLab_dayOfWeek', date_format('OsBuildLab_date', 'u'))\
.withColumn('AvSigVersion_dayOfWeek', date_format('DateAvSigVersion', 'u'))

final_dates.persist()
final_dates.count()

drop_list = ['Platform', 'OsBuildLab_date', 'DateAvSigVersion', 'OsBuildLab_date_lag',
             'DateAvSigVersion_lag', 'max(OsBuildLab_date)', 'max(DateAvSigVersion)',
             'OsBuildLab_date_fulllag', 'DateAvSigVersion_fulllag', 'max(OSBuild_diff)',
             'max(AvSigVersion_diff)', 'max(OSBuild_fulldiff)', 'max(AvSigVersion_fulldiff)']

for c in drop_list:
    final_dates = final_dates.drop(c)

final_dates_imputed = final_dates.fillna(0)

write_path = '../../../data/df_dates_0'
print('Guardamos el DF en {}'.format(write_path))
final_dates_imputed.write.csv(write_path, sep=',', mode="overwrite", header=True)
