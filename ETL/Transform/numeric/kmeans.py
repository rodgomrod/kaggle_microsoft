from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

spark = SparkSession.builder.appName('RF_trainer').getOrCreate()

df_num = spark.read \
    .options(header = "true", sep=',', inferschema = "true") \
    .csv('data/df_num_imputed_3/*.csv')

print('DF leido')

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

print('Convertimos el DF numerico a vector')

assembler_features = VectorAssembler(inputCols=continuous_columns, outputCol='features')
train_data = assembler_features.transform(df_num)
train_data_final = train_data.select('features', 'MachineIdentifier')

print('Persistimos el DF vector')
train_data_final.persist()
print(train_data_final.count())

# Aqui guardaremos los resultados
df_kmeans = train_data_final.select('MachineIdentifier')

print('Empezamos a entrenar los kmeans')
ks = [2, 4, 8, 16, 32, 64]
for k in ks:
    print('KMeans con k = {}'.format(k))
    kmeans = KMeans(predictionCol='prediction_{}'.format(k),
                    featuresCol='features').setK(k).setSeed(1)

    model = kmeans.fit(train_data_final)
    df_tra = model.transform(train_data_final)
    df_kmeans = df_kmeans.join(df_tra.select('MachineIdentifier',
                                             'prediction_{}'.format(k)),
                               ['MachineIdentifier'])


write_path = 'data/df_kmeans_2'

print('Guardando CSV en {}'.format(write_path))
df_kmeans.write.csv(write_path, sep=',', mode="overwrite", header=True)