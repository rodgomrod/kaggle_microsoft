from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, IntegerType, LongType)
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

# Inicio de la sesion de Spark
spark = SparkSession.builder.appName('separador_cat_num').getOrCreate()

# Cargando train y test con inferSchema = True
df_train = spark.read.csv('data/train.csv', header=True, inferSchema=True)
df_test = spark.read.csv('data/test.csv', header=True, inferSchema=True)

# Añadimos la columna "HasDetections" al test
df_test = df_test.withColumn('HasDetections', lit(None))

# Unimos train y test en un unico DF

full_df = df_train.union(df_test)

full_df.persist()
print('Nº de filas en full_df', full_df.count())

# Separamos en columnas categoricas y numericas
cat_cols = list()
num_cols = list()
all_types = full_df.dtypes
for col in all_types:
    if col[1] == 'string':
        cat_cols.append(col[0])
    else:
        num_cols.append(col[0])

df_cat = full_df.select(cat_cols)
# Añadimos tambien "MachineIdentifier" para poder hacer un inner join posterior
df_num = full_df.select(num_cols+['MachineIdentifier'])

cat_write_path = 'data/df_cat'
num_write_path = 'data/df_num'

print('Guardando CSV numerico y categorico')
df_cat.write.csv(cat_write_path, sep=',', mode="overwrite", header=True)
df_num.write.csv(num_write_path, sep=',', mode="overwrite", header=True)

