from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, IntegerType, LongType)
from pyspark.sql.functions import *
from pyspark import SparkConf
from pyspark import SparkContext
import multiprocessing

# Configuracion de memoria y cores
cores = multiprocessing.cpu_count()
p = 20
conf = SparkConf()
conf.set("spark.driver.cores", cores)
conf.set("spark.driver.memory", "10g")
conf.set("spark.sql.shuffle.partitions", p * cores)
conf.set("spark.default.parallelism", p * cores)
sc = SparkContext(conf=conf)

# SparkSession
spark = SparkSession.builder.appName("Microsoft_Kaggle").getOrCreate()

# Read data
data = spark.read.csv('data/df_cat/*.csv', header=True, inferSchema=True)

# Persistimos el DF para mejorar el rendimiento
data.persist()
print('Numero de casos totales = {}'.format(data.count()))

init_cols = data.columns


# Transformaciones

## ProductName
	# Label enconding para variables categoricas
indexer = StringIndexer(inputCol="ProductName", outputCol="ProductNameIndex")
data = indexer.fit(data).transform(data)


## Census_PrimaryDiskTypeName
	# Label encoding para Census_PrimaryDiskTypeName.
data = data.fillna( { 'Census_PrimaryDiskTypeName':'UNKNOWN'} )
indexer = StringIndexer(inputCol="Census_PrimaryDiskTypeName", outputCol="Census_PrimaryDiskTypeNameIndex")
data = indexer.fit(data).transform(data)


## Census_ChassisTypeName
	# Frecuencia 
frequency_census = data.groupBy('Census_ChassisTypeName').count().withColumnRenamed('count','Census_ChassisTypeName_freq')
data = data.join(frequency_census,'Census_ChassisTypeName','left')


## Census_PowerPlatformRoleName
	# Label encoding
data = data.fillna( { 'Census_PowerPlatformRoleName':'UNKNOWN'} )
indexer = StringIndexer(inputCol="Census_PowerPlatformRoleName", outputCol="Census_PowerPlatformRoleNameIndex")
data = indexer.fit(data).transform(data)

## Census_InternalBatteryType
	# Frecuencia
frequency_census = data.groupBy('Census_InternalBatteryType').count().withColumnRenamed('count','Census_InternalBatteryType_freq')
data = data.join(frequency_census,'Census_InternalBatteryType','left')

	#Booleana
data = data.withColumn('Census_InternalBatteryType_informed',when(col('Census_InternalBatteryType').isNotNull(),1).otherwise(0))


## Census_OSVersion 


## Census_OSArchitecture 
	# Label enconding clarisimo

indexer = StringIndexer(inputCol="Census_OSArchitecture", outputCol="Census_OSArchitectureIndex")
data = indexer.fit(data).transform(data)


## Census_OSBranch 
	# frequency 
frequency_census = data.groupBy('Census_OSBranch').count().withColumnRenamed('count','Census_OSBranch_freq')
data = data.join(frequency_census,'Census_OSBranch','left')


#######

# Persist intermedio
data.persist()
data.first()

## EngineVersion
	# Al ser una version, se ha hecho split por el punto "."
    # [0] y [1] es igual para el DF al completo, se ignora
data = data.withColumn('EngineVersion_2', split(data['EngineVersion'], '\.')[2].cast(IntegerType()))\
.withColumn('EngineVersion_3', split(data['EngineVersion'], '\.')[3].cast(IntegerType()))


## AppVersion
	# Al ser una version, se ha hecho split por el punto "."
    # [0] es igual para el DF al completo, se ignora
data = data.withColumn('AppVersion_1', split(data['AppVersion'], '\.')[1].cast(IntegerType()))\
.withColumn('AppVersion_2', split(data['AppVersion'], '\.')[2].cast(IntegerType()))\
.withColumn('AppVersion_3', split(data['AppVersion'], '\.')[3].cast(IntegerType()))


## AvSigVersion
	# Al ser una version, se ha hecho split por el punto "."
    # [3] es igual para el DF al completo, se ignora
data = data.withColumn('AvSigVersion_0', split(data['AvSigVersion'], '\.')[0].cast(IntegerType()))\
.withColumn('AvSigVersion_1', split(data['AvSigVersion'], '\.')[1].cast(IntegerType()))\
.withColumn('AvSigVersion_2', split(data['AvSigVersion'], '\.')[2].cast(IntegerType()))


## Platform
	# Label enconding para variables categoricas
indexer_Platform = StringIndexer(inputCol="Platform", outputCol="PlatformIndex")
data = indexer_Platform.fit(data).transform(data)


## Processor
	# Label enconding para variables categoricas
indexer_processor = StringIndexer(inputCol="Processor", outputCol="ProcessorIndex")
data = indexer_processor.fit(data).transform(data)


## OsVer
	# Frecuencia
df_cat_freq_osver = data.groupBy('OsVer').count().withColumnRenamed('count', 'OsVer_freq')
data = data.join(df_cat_freq_osver, ['OsVer'], 'left')


## OsPlatformSubRelease
	# Label enconding para variables categoricas
indexer_OsPlatformSubRelease = StringIndexer(inputCol="OsPlatformSubRelease", outputCol="OsPlatformSubReleaseIndex")
data = indexer_OsPlatformSubRelease.fit(data).transform(data)


## OsBuildLab
	# split por punto "." y transformamos
data = data.withColumn('OsBuildLab_0', split(data['OsBuildLab'], '\.')[0].cast(IntegerType()))\
.withColumn('OsBuildLab_1', split(data['OsBuildLab'], '\.')[1].cast(IntegerType()))\
.withColumn('OsBuildLab_2', split(data['OsBuildLab'], '\.')[2])\
.withColumn('OsBuildLab_3', split(data['OsBuildLab'], '\.')[3])\
.withColumn('OsBuildLab_4', split(data['OsBuildLab'], '\.')[4])\
.withColumn('OsBuildLab_4_0', split(data['OsBuildLab_4'], '-')[0].cast(IntegerType()))\
.withColumn('OsBuildLab_4_1', split(data['OsBuildLab_4'], '-')[1].cast(IntegerType()))


## SkuEdition
	# Label enconding para variables categoricas
indexer_SkuEdition = StringIndexer(inputCol="SkuEdition", outputCol="SkuEditionIndex")
data = indexer_SkuEdition.fit(data).transform(data)


## PuaMode
	# Label enconding para variables categoricas
data = data.withColumn('PuaMode', when(col('PuaMode').isNull(), 'UNKNOWN').otherwise(col('PuaMode')))
indexer_PuaMode = StringIndexer(inputCol="PuaMode", outputCol="PuaModeIndex")
data = indexer_PuaMode.fit(data).transform(data)


## SmartScreen
	# Frecuencia
df_cat_freq_SmartScreen = data.groupBy('SmartScreen').count().withColumnRenamed('count', 'SmartScreen_freq')
data = data.join(df_cat_freq_SmartScreen, ['SmartScreen'], 'left')


## Census_MDC2FormFactor
	# Frecuencia
df_cat_freq_osver = data.groupBy('Census_MDC2FormFactor').count().withColumnRenamed('count', 'Census_MDC2FormFactor_freq')
data = data.join(df_cat_freq_osver, ['Census_MDC2FormFactor'], 'left')


## Census_DeviceFamily
	# Label enconding para variables categoricas
indexer_Census_DeviceFamily = StringIndexer(inputCol="Census_DeviceFamily", outputCol="Census_DeviceFamilyIndex")
data = indexer_Census_DeviceFamily.fit(data).transform(data)




# Guardamos el DF con las variables categoricas transformadas
final_cols = data.columns
cols_transformadas = list(set(final_cols) - set(init_cols))

write_path = 'data/df_cat_transform_0'
final_data = data.select(['MachineIdentifier'] + cols_transformadas)
final_data.write.csv(write_path, sep=',', mode="overwrite", header=True)



