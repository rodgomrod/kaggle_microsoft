from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from pyspark import SparkConf
from pyspark import SparkContext
import multiprocessing

# =============================================================================
# Configuracion de memoria y nº particiones
# =============================================================================
cores = multiprocessing.cpu_count()
p = 2
conf = SparkConf()
conf.set("spark.driver.cores", cores)
conf.set("spark.driver.memory", "55g")
conf.set("spark.sql.shuffle.partitions", p * cores)
conf.set("spark.default.parallelism", p * cores)
sc = SparkContext(conf=conf)

# SparkSession
spark = SparkSession.builder.appName('MK_genera_train_test').getOrCreate()

df_num = spark.read.csv('data/df_num_imputed_0/*.csv', header=True, inferSchema=True)
df_num.persist()
df_num.count()

df_cat = spark.read.csv('data/df_cat_pro_0/*.csv', header=True, inferSchema=True)
df_cat.persist()
df_cat.count()

full_df = df_num.join(df_cat, ['MachineIdentifier'])

train = full_df.filter(col('HasDetections').isNotNull())
train = train.fillna(-99)
test = full_df.filter(col('HasDetections').isNull())
test = test.fillna(-99)

write_path_train = 'data/train_final_1'
write_path_test = 'data/test_final_1'
train.coalesce(1).write.csv(write_path_train, sep=',', mode="overwrite", header=True)
test.coalesce(1).write.csv(write_path_test, sep=',', mode="overwrite", header=True)
