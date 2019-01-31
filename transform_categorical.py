from pyspark.sql import SparkSession


from pyspark.ml.feature import StringIndexer

spark = SparkSession.builder.appName("Microsoft_Kaggle").getOrCreate()

# Read data

data = #Todo


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



#######

# Rodrigo



