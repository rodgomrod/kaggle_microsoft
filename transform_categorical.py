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