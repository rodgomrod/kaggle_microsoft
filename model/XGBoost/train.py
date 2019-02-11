import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars ../../jars/xgboost4j-spark-0.72.jar,../../jars/xgboost4j-0.72.jar pyspark-shell'
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier

spark = SparkSession.builder.appName('XGB_trainer').getOrCreate()

spark.sparkContext.addPyFile("/home/ayesa1/Descargas/sparkxgb.zip")
from sparkxgb import XGBoostEstimator

spark = SparkSession.builder.appName('RF_trainer').getOrCreate()

path_train = '../../data/train_final_0'
print('Carga del TRAIN', path_train)
train = spark.read \
    .options(header = "true", sep=',', inferschema = "true") \
    .csv(path_train)

train.persist()
print("Numero de casos en el train: %d" % train.count())

ignore_c = ['MachineIdentifier', 'HasDetections']
train_cols = [c for c in train.columns if c not in ignore_c]

# Convertimos el TRAIN en un VECTOR para poder pasarle el RF
print('Conversion de datos a VectorAssembler')
assembler_features = VectorAssembler(inputCols=train_cols, outputCol='features')
train_data = assembler_features.transform(train)
train_data = train_data.select('features', 'HasDetections')


xgboost = XGBoostEstimator(
    featuresCol="features",
    labelCol="HasDetections",
    predictionCol="prediction"
)

pipeline = Pipeline().setStages([xgboost])

trainDF, testDF = train_data.randomSplit([0.8, 0.2], seed=24)

model = pipeline.fit(trainDF)
preds = model.transform(testDF)

preds.select(col("HasDetections"), col("prediction")).show()
preds.show()
