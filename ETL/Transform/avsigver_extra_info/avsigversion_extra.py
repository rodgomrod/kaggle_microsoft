from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, IntegerType, LongType)
from pyspark.sql.functions import *
from pyspark.ml import Pipeline

import multiprocessing


print('Inicio del Script\n')



spark = SparkSession.builder.appName("Microsoft_Kaggle").getOrCreate()
# spark.conf.set("spark.sql.shuffle.partitions", p * cores)
# spark.conf.set("spark.default.parallelism", p * cores)

data = spark.read.csv('Notebooks/new_variables/AVSigversion_Threats/AvSigversion_Threats.csv', header=True, inferSchema=True)

data = data.withColumnRenamed('Name','AvSigVersion_Name')\
        .withColumnRenamed('AlertLevel','AvSigVersion_AlertLevel')\
        .withColumnRenamed('Type','AvSigVersion_Type')
data = data.drop('index')

df_num = spark.read.csv("data/df_cat_prepro_0/*.csv",inferSchema=True,header=True).select('MachineIdentifier','AvSigVersion')



data = data.withColumn('AvSigVersion_Name_1', split(data['AvSigVersion_Name'], ':')[0])\
            .withColumn('AvSigVersion_Name_2', split(data['AvSigVersion_Name'], '/')[0])


cols_label_encoder = ['AvSigVersion_AlertLevel','AvSigVersion_Type','AvSigVersion_Name_1','AvSigVersion_Name_2']


print('Pipeline de Indexers paras las columnas {0}\n'.format(cols_label_encoder))
indexers = [StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid="keep").fit(data) for c in cols_label_encoder]
pipeline = Pipeline(stages=indexers)
data = pipeline.fit(data).transform(data)



df_save = df_num.join(data,'AvSigVersion','left')


columns_for_save = ['MachineIdentifier','AvSigVersion_AlertLevel_index','AvSigVersion_Type_index','AvSigVersion_Name_1_index','AvSigVersion_Name_2_index']
write_path = 'data/df_avsig_version'

print('Guardamos el DF en {}'.format(write_path))
df_save.select(columns_for_save).write.csv(write_path,sep=',', mode="overwrite", header=True)

