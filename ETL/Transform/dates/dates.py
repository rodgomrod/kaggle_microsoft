from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, IntegerType, LongType, DateType)

spark = SparkSession.builder.appName("Microsoft_Kaggle").getOrCreate()

df_num = spark.read.csv("data/df_cat_prepro_0/*.csv", header=True, inferSchema=True)\
    .select('MachineIdentifier', 'OsBuildLab', 'AvSigVersion', 'Census_OSVersion', 'OsPlatformSubRelease')

print('DF leido')

df_fechas_av = spark.read.csv("Notebooks/new_variables/fechas_av.csv", inferSchema=True, header=True)
df_fechas_os = spark.read.csv("Notebooks/new_variables/fechas_os.csv", inferSchema=True, header=True)

df_fechas_os = df_fechas_os.withColumn('DateOSVersion', to_date(col('DateCensus_OSVersion')))
df_fechas_av = df_fechas_av.withColumn('DateAvSigVersion', to_date(col('DateAvSigVersion')))

print('Fechas leidas')

df_date_osbuild = df_num.withColumn('OsBuildLab_4', split(df_num['OsBuildLab'], '\.')[4])
df_date_osbuild = df_date_osbuild.withColumn('OsBuildLab_date', split(df_date_osbuild['OsBuildLab_4'], '-')[0])

df_date_osbuild = df_date_osbuild.withColumn('DateOsBuildLab', to_date(col('OsBuildLab_date'), format='yyMMdd'))

df_dates = df_date_osbuild.join(df_fechas_av, ['AvSigVersion'], 'left').select('MachineIdentifier',
                                                                               'DateOsBuildLab',
                                                                               'DateAvSigVersion',
                                                                               'Census_OSVersion',
                                                                               'OsPlatformSubRelease')


df_dates = df_dates.join(df_fechas_os, ['Census_OSVersion'], 'left').select('MachineIdentifier',
                                                                            'DateOsBuildLab',
                                                                            'DateAvSigVersion',
                                                                            'DateOSVersion',
                                                                            'OsPlatformSubRelease')

df_dates.persist()
df_dates.count()

# Funcion para el calculo del STD rapido
def pySparkSTD(x, y, z):
    if x == None:
        x = 0
    if y == None:
        y = 0
    if z == None:
        z = 0
    med = (x + y + z)/3
    return ((((x-med)**2) + ((y-med)**2) + ((z-med)**2))/2)**(1/2)


udf_std = udf(pySparkSTD, DoubleType())

w1 = Window.partitionBy('OsPlatformSubRelease').orderBy('DateOsBuildLab')
w2 = Window.partitionBy('OsPlatformSubRelease').orderBy('DateAvSigVersion')
w3 = Window.partitionBy('OsPlatformSubRelease').orderBy('DateOSVersion')

print("window 1")
data_windows = df_dates.withColumn('DateOsBuildLab_lag', lag('DateOsBuildLab').over(w1))
data_windows = data_windows.withColumn('OsBuildLab_diff', datediff(col('DateOsBuildLab'), col('DateOsBuildLab_lag')))
data_windows = data_windows.withColumn('OsBuildLab_diff_lead', lead('OsBuildLab_diff').over(w1))
data_windows = data_windows.withColumn('OsBuildLab_diff_lag', lag('OsBuildLab_diff').over(w1))
data_windows = data_windows.withColumn('std_diff_DateOsBuildLab', udf_std('OsBuildLab_diff', 'OsBuildLab_diff_lead', 'OsBuildLab_diff_lag'))
# print(data_windows.show())

print("window 2")
data_windows1 = data_windows.withColumn('DateAvSigVersion_lag', lag('DateAvSigVersion').over(w2))
data_windows1 = data_windows1.withColumn('AvSigVersion_diff', datediff(col('DateAvSigVersion'), col('DateAvSigVersion_lag')))
data_windows1 = data_windows1.withColumn('AvSigVersion_diff_lead', lead('AvSigVersion_diff').over(w2))
data_windows1 = data_windows1.withColumn('AvSigVersion_diff_lag', lag('AvSigVersion_diff').over(w2))
data_windows1 = data_windows1.withColumn('std_diff_AvSigVersion', udf_std('AvSigVersion_diff', 'AvSigVersion_diff_lead', 'AvSigVersion_diff_lag'))
# print(data_windows1.show())

print("window 3")
data_windows2 = data_windows1.withColumn('DateOSVersion_lag', lag('DateOSVersion').over(w3))
data_windows2 = data_windows2.withColumn('OSVersion_diff', datediff(col('DateOSVersion'), col('DateOSVersion_lag')))
data_windows2 = data_windows2.withColumn('OSVersion_diff_lead', lead('OSVersion_diff').over(w3))
data_windows2 = data_windows2.withColumn('OSVersion_diff_lag', lag('OSVersion_diff').over(w3))
data_windows2 = data_windows2.withColumn('std_diff_OSVersion', udf_std('OSVersion_diff', 'OSVersion_diff_lead', 'OSVersion_diff_lag'))


w4 = Window.partitionBy('DateOsBuildLab')
w5 = Window.partitionBy('DateAvSigVersion')
w6 = Window.partitionBy('DateOSVersion')

df_max_diff = data_windows2.withColumn('max_OsBuildLab_diff', max('OsBuildLab_diff').over(w4))\
                    .withColumn('max_AvSigVersion_diff', max('AvSigVersion_diff').over(w5))\
                    .withColumn('max_OSVersion_diff', max('OSVersion_diff').over(w6))


print("ultimo join")

df_max_diff_ratios = df_max_diff.withColumn('ratio_OsBuildLab_diff', col('max_OsBuildLab_diff')/col('OsBuildLab_diff'))\
    .withColumn('ratio_AvSigVersion_diff', col('max_AvSigVersion_diff')/col('AvSigVersion_diff'))\
    .withColumn('ratio_OSVersion_diff', col('max_OSVersion_diff')/col('OSVersion_diff'))


drop_list = [
    'DateOSVersion_lag',
    'OSVersion_diff_lead',
    'OSVersion_diff_lag',
    'DateOsBuildLab_lag',
    'OsBuildLab_diff_lead',
    'OsBuildLab_diff_lag',
    'DateAvSigVersion_lag',
    'AvSigVersion_diff_lead',
    'AvSigVersion_diff_lag',
    'DateOsBuildLab',
    'DateAvSigVersion',
    'DateOSVersion',
    'OsPlatformSubRelease'
]

final_dates = df_max_diff_ratios.drop(*drop_list).fillna(0)

write_path = 'data/df_dates_3'
print('Guardamos el DF en {}'.format(write_path))
final_dates.write.csv(write_path, sep=',', mode="overwrite", header=True)
