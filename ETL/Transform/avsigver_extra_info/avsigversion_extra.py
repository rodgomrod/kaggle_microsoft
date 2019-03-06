from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, IntegerType, LongType)
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.sql.window import Window

import multiprocessing


print('Inicio del Script\n')



spark = SparkSession.builder.appName("Microsoft_Kaggle").getOrCreate()

data = spark.read.csv('Notebooks/new_variables/AVSigversion_Threats/AvSigversion_Threats.csv', header=True, inferSchema=True)

data = data.withColumnRenamed('Name','AvSigVersion_Name')\
        .withColumnRenamed('AlertLevel','AvSigVersion_AlertLevel')\
        .withColumnRenamed('Type','AvSigVersion_Type')
data = data.drop('index')

data_pro = data.groupBy('AvSigVersion')\
    .agg(countDistinct('AvSigVersion_Name'),
         countDistinct('AvSigVersion_Type'),
         countDistinct('AvSigVersion_AlertLevel'))

# w = Window.partitionBy('AvSigVersion')
#
# a = data.withColumn("count", count("AvSigVersion_AlertLevel").over(w)).orderBy("count", ascending=False)\
#     .groupBy("AvSigVersion")\
#     .agg(count("count").alias("AvSigVersion_Name_value"))

df_num = spark.read.csv("data/df_cat_prepro_0/*.csv", header=True).select('MachineIdentifier','AvSigVersion')

df_save = df_num.join(data_pro,'AvSigVersion', 'left_outer')


columns_for_save = ['MachineIdentifier','count(DISTINCT AvSigVersion_Name)','count(DISTINCT AvSigVersion_Type)',
                    'count(DISTINCT AvSigVersion_AlertLevel)']
write_path = 'data/df_avsig_version'

print('Guardamos el DF en {}'.format(write_path))
df_save.select(columns_for_save).write.csv(write_path,sep=',', mode="overwrite", header=True)

