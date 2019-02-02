from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, IntegerType, LongType)
from pyspark.sql.functions import *
from pyspark import SparkConf
from pyspark import SparkContext
import multiprocessing
from pyspark.ml import Pipeline

print('Inicio del Script')

# Configuracion de memoria y cores
cores = multiprocessing.cpu_count()
p = 3
particiones = cores * p
memoria = 16 # memoria ram instalada
dm = memoria/2
conf = SparkConf()
conf.set("spark.driver.cores", cores)
conf.set("spark.driver.memory", "13g")
conf.set("spark.sql.shuffle.partitions", particiones)
conf.set("spark.default.parallelism", particiones)
sc = SparkContext(conf=conf)

# SparkSession
spark = SparkSession.builder.appName("Microsoft_Kaggle").getOrCreate()

# Read data
print('Lectura del DF crudo')
data = spark.read.csv('data/df_cat/*.csv', header=True, inferSchema=True)

# Persistimos el DF para mejorar el rendimiento
data.persist()
print('Numero de casos totales = {}'.format(data.count()))

init_cols = data.columns


# Transformaciones
print('Inicio de las transformaciones:')

## Indexers
    # Label enconding para variables categoricas
columnas_indexer = ['ProductName', 'Census_PrimaryDiskTypeName', 'Census_PowerPlatformRoleName', 'Census_OSArchitecture',
                    'Census_ProcessorClass', 'Census_OSInstallTypeName', 'Census_OSWUAutoUpdateOptionsName',
                    'Census_GenuineStateName', 'Platform', 'Processor', 'OsPlatformSubRelease', 'SkuEdition', 'PuaMode',
                    'Census_DeviceFamily']
print('\tIndexers paras las columnas {}'.format(columnas_indexer))
indexers = [StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid="keep").fit(data) for c in columnas_indexer]
pipeline = Pipeline(stages=indexers)
data0 = pipeline.fit(data).transform(data)

# Imputamos los nulls que hayan quedado
imputaciones = dict()
for c in columnas_indexer:
    imputaciones[c] = -1
data = data0.fillna(imputaciones)

# Persist intermedio
print('Persist intermedio')
data.persist()
data.first()

# print('\tProductName')
# ## ProductName
# 	# Label enconding para variables categoricas
# indexer = StringIndexer(inputCol="ProductName", outputCol="ProductNameIndex")
# data = indexer.fit(data).transform(data)

# print('\tCensus_PrimaryDiskTypeName')
# ## Census_PrimaryDiskTypeName
# 	# Label encoding para Census_PrimaryDiskTypeName.
# data = data.fillna( { 'Census_PrimaryDiskTypeName':'UNKNOWN'} )
# indexer = StringIndexer(inputCol="Census_PrimaryDiskTypeName", outputCol="Census_PrimaryDiskTypeNameIndex")
# data = indexer.fit(data).transform(data)

print('\tCensus_ChassisTypeName')
## Census_ChassisTypeName
	# Frecuencia 
frequency_census = data.groupBy('Census_ChassisTypeName').count().withColumnRenamed('count','Census_ChassisTypeName_freq')
data = data.join(frequency_census,'Census_ChassisTypeName','left')

# print('\tCensus_PowerPlatformRoleName')
# ## Census_PowerPlatformRoleName
# 	# Label encoding
# # data = data.fillna( { 'Census_PowerPlatformRoleName':'UNKNOWN'} )
# indexer = StringIndexer(inputCol="Census_PowerPlatformRoleName", outputCol="Census_PowerPlatformRoleNameIndex")
# data = indexer.fit(data).transform(data)

print('\tCensus_InternalBatteryType')
## Census_InternalBatteryType
	# Frecuencia
frequency_census = data.groupBy('Census_InternalBatteryType').count().withColumnRenamed('count','Census_InternalBatteryType_freq')
data = data.join(frequency_census,'Census_InternalBatteryType','left')

	#Booleana
data = data.withColumn('Census_InternalBatteryType_informed', when(col('Census_InternalBatteryType').isNotNull(),1).otherwise(0))

print('\tCensus_OSVersion')
## Census_OSVersion
    # Al ser una version, se ha hecho split por el punto "."
data = data.withColumn('Census_OSVersion_0', split(data['Census_OSVersion'], '\.')[0].cast(IntegerType()))\
.withColumn('Census_OSVersion_1', split(data['Census_OSVersion'], '\.')[1].cast(IntegerType()))\
.withColumn('Census_OSVersion_2', split(data['Census_OSVersion'], '\.')[2].cast(IntegerType()))\
.withColumn('Census_OSVersion_3', split(data['Census_OSVersion'], '\.')[3].cast(IntegerType()))

# print('\tCensus_OSArchitecture')
# ## Census_OSArchitecture
# 	# Label enconding clarisimo
# indexer = StringIndexer(inputCol="Census_OSArchitecture", outputCol="Census_OSArchitectureIndex")
# data = indexer.fit(data).transform(data)

print('\tCensus_OSBranch')
## Census_OSBranch 
	# frequency 
frequency_census = data.groupBy('Census_OSBranch').count().withColumnRenamed('count','Census_OSBranch_freq')
data = data.join(frequency_census,'Census_OSBranch','left')

# print('\tCensus_ProcessorClass')
# ## Census_ProcessorClass
# 	# Label enconding para variables categoricas
# # data = data.withColumn('Census_ProcessorClass', when(col('Census_ProcessorClass').isNull(), 'UNKNOWN').otherwise(col('Census_ProcessorClass')))
# indexer_Census_ProcessorClass = StringIndexer(inputCol="Census_ProcessorClass", outputCol="Census_ProcessorClassIndex")
# data = indexer_Census_ProcessorClass.fit(data).transform(data)

print('\tCensus_OSEdition')
## Census_OSEdition
	# Frecuencia
df_cat_freq_Census_OSEdition = data.groupBy('Census_OSEdition').count().withColumnRenamed('count', 'Census_OSEdition_freq')
data = data.join(df_cat_freq_Census_OSEdition, ['Census_OSEdition'], 'left')


print('\tCensus_OSSkuName')
## Census_OSSkuName
	# Frecuencia
frequency_Census_OSSkuName = data.groupBy('Census_OSSkuName').count().withColumnRenamed('count','Census_OSSkuName_freq')
data = data.join(frequency_Census_OSSkuName,'Census_OSSkuName','left')

# print('\tCensus_OSInstallTypeName')
# ## Census_OSInstallTypeName
# 	# Label enconding para variables categoricas
# indexer_Census_OSInstallTypeName = StringIndexer(inputCol="Census_OSInstallTypeName", outputCol="Census_OSInstallTypeNameIndex")
# data = indexer_Census_OSInstallTypeName.fit(data).transform(data)

# print('\tCensus_OSWUAutoUpdateOptionsName')
# ## Census_OSWUAutoUpdateOptionsName
# 	# Label enconding para variables categoricas
# indexer_Census_OSWUAutoUpdateOptionsName = StringIndexer(inputCol="Census_OSWUAutoUpdateOptionsName", outputCol="Census_OSWUAutoUpdateOptionsNameIndex")
# data = indexer_Census_OSWUAutoUpdateOptionsName.fit(data).transform(data)

# print('\tCensus_GenuineStateName')
# ## Census_GenuineStateName
# 	# Label enconding para variables categoricas
# indexer_Census_GenuineStateName = StringIndexer(inputCol="Census_GenuineStateName", outputCol="Census_GenuineStateNameIndex")
# data = indexer_Census_GenuineStateName.fit(data).transform(data)

# print('\tCensus_GenuineStateName')
# ## Census_GenuineStateName
# 	# Label enconding para variables categoricas
# indexer_Census_ActivationChannel = StringIndexer(inputCol="Census_ActivationChannel", outputCol="Census_ActivationChannelIndex")
# data = indexer_Census_ActivationChannel.fit(data).transform(data)

print('\tCensus_FlightRing')
## Census_FlightRing
	# Frecuencia
frequency_Census_Census_FlightRing = data.groupBy('Census_FlightRing').count().withColumnRenamed('count','Census_FlightRing_freq')
data = data.join(frequency_Census_Census_FlightRing,'Census_FlightRing','left')


#######

print('\tEngineVersion')
## EngineVersion
	# Al ser una version, se ha hecho split por el punto "."
    # [0] y [1] es igual para el DF al completo, se ignora
data = data.withColumn('EngineVersion_2', split(data['EngineVersion'], '\.')[2].cast(IntegerType()))\
.withColumn('EngineVersion_3', split(data['EngineVersion'], '\.')[3].cast(IntegerType()))

print('\tAppVersion')
## AppVersion
	# Al ser una version, se ha hecho split por el punto "."
    # [0] es igual para el DF al completo, se ignora
data = data.withColumn('AppVersion_1', split(data['AppVersion'], '\.')[1].cast(IntegerType()))\
.withColumn('AppVersion_2', split(data['AppVersion'], '\.')[2].cast(IntegerType()))\
.withColumn('AppVersion_3', split(data['AppVersion'], '\.')[3].cast(IntegerType()))

print('\tAvSigVersion')
## AvSigVersion
	# Al ser una version, se ha hecho split por el punto "."
    # [3] es igual para el DF al completo, se ignora
data = data.withColumn('AvSigVersion_0', split(data['AvSigVersion'], '\.')[0].cast(IntegerType()))\
.withColumn('AvSigVersion_1', split(data['AvSigVersion'], '\.')[1].cast(IntegerType()))\
.withColumn('AvSigVersion_2', split(data['AvSigVersion'], '\.')[2].cast(IntegerType()))

# print('\tPlatform')
# ## Platform
# 	# Label enconding para variables categoricas
# indexer_Platform = StringIndexer(inputCol="Platform", outputCol="PlatformIndex")
# data = indexer_Platform.fit(data).transform(data)

# print('\tProcessor')
# ## Processor
# 	# Label enconding para variables categoricas
# indexer_processor = StringIndexer(inputCol="Processor", outputCol="ProcessorIndex")
# data = indexer_processor.fit(data).transform(data)

print('\tOsVer')
## OsVer
	# Frecuencia
df_cat_freq_osver = data.groupBy('OsVer').count().withColumnRenamed('count', 'OsVer_freq')
data = data.join(df_cat_freq_osver, ['OsVer'], 'left')

# print('\tOsPlatformSubRelease')
# ## OsPlatformSubRelease
# 	# Label enconding para variables categoricas
# indexer_OsPlatformSubRelease = StringIndexer(inputCol="OsPlatformSubRelease", outputCol="OsPlatformSubReleaseIndex")
# data = indexer_OsPlatformSubRelease.fit(data).transform(data)

print('\tOsBuildLab')
## OsBuildLab
	# split por punto "." y transformamos
data1 = data.withColumn('OsBuildLab_0', split(data['OsBuildLab'], '\.')[0].cast(IntegerType()))\
.withColumn('OsBuildLab_1', split(data['OsBuildLab'], '\.')[1].cast(IntegerType()))\
.withColumn('OsBuildLab_2', split(data['OsBuildLab'], '\.')[2])\
.withColumn('OsBuildLab_3', split(data['OsBuildLab'], '\.')[3])\
.withColumn('OsBuildLab_4', split(data['OsBuildLab'], '\.')[4])

data = data1.withColumn('OsBuildLab_4_0', split(data1['OsBuildLab_4'], '-')[0].cast(IntegerType()))\
.withColumn('OsBuildLab_4_1', split(data1['OsBuildLab_4'], '-')[1].cast(IntegerType()))

# print('\tSkuEdition')
# ## SkuEdition
# 	# Label enconding para variables categoricas
# indexer_SkuEdition = StringIndexer(inputCol="SkuEdition", outputCol="SkuEditionIndex")
# data = indexer_SkuEdition.fit(data).transform(data)

# print('\tPuaMode')
# ## PuaMode
# 	# Label enconding para variables categoricas
# # data = data.withColumn('PuaMode', when(col('PuaMode').isNull(), 'UNKNOWN').otherwise(col('PuaMode')))
# indexer_PuaMode = StringIndexer(inputCol="PuaMode", outputCol="PuaModeIndex")
# data = indexer_PuaMode.fit(data).transform(data)

print('\tSmartScreen')
## SmartScreen
	# Frecuencia
df_cat_freq_SmartScreen = data.groupBy('SmartScreen').count().withColumnRenamed('count', 'SmartScreen_freq')
data = data.join(df_cat_freq_SmartScreen, ['SmartScreen'], 'left')

print('\tCensus_MDC2FormFactor')
## Census_MDC2FormFactor
	# Frecuencia
df_cat_freq_osver = data.groupBy('Census_MDC2FormFactor').count().withColumnRenamed('count', 'Census_MDC2FormFactor_freq')
data = data.join(df_cat_freq_osver, ['Census_MDC2FormFactor'], 'left')

# print('\tCensus_DeviceFamily')
# ## Census_DeviceFamily
# 	# Label enconding para variables categoricas
# indexer_Census_DeviceFamily = StringIndexer(inputCol="Census_DeviceFamily", outputCol="Census_DeviceFamilyIndex")
# data = indexer_Census_DeviceFamily.fit(data).transform(data)



# Guardamos el DF con las variables categoricas transformadas
final_cols = data.columns
cols_transformadas = list(set(final_cols) - set(init_cols))

write_path = 'data/df_cat_transform_0'
print('Guardamos el DF en {}'.format(write_path))
final_data = data.select(['MachineIdentifier'] + cols_transformadas)
final_data.write.csv(write_path, sep=',', mode="overwrite", header=True)



