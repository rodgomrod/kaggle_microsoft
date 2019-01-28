from pyspark.sql import SparkSession


from pyspark.ml.feature import StringIndexer
import lib.utilies as utils


spark = SparkSession.builder.appName("Microsoft_Kaggle").getOrCreate()

# Read data

data = #Todo


# FillNA
data = data.fillna( { 'Census_PrimaryDiskTypeName':'UNKNOWN'} )

# Transformaciones
to_label_enconding = ['ProductName','Census_PrimaryDiskTypeName']

for i in to_label_enconding:
	utils.label_encoding_col(data,i)




## Census_ChassisTypeName
	# Frecuencia 
frequency_census = data.groupBy('Census_ChassisTypeName').count().withColumnRenamed('count','Census_ChassisTypeName_freq')
data = data.join(frequency_census,'Census_ChassisTypeName','left')





#######

# Rodrigo



