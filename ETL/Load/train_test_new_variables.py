from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# SparkSession
spark = SparkSession.builder.appName('MK_genera_train_test').getOrCreate()

df_num = spark.read.csv('data/df_num_imputed_3/*.csv', header=True, inferSchema=True)
# df_num.persist()
# df_num.count()


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
