from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np
import functools
from pyspark.ml.classification import RandomForestClassifier

spark = SparkSession.builder.appName('RF_trainer').getOrCreate()

path_train = '../../data/train_final_0'
print('Carga del TRAIN', path_train)
train = spark.read \
    .options(header = "true", sep=',', inferschema = "true") \
    .csv(path_train)

train.persist()
print("Numero de casos en el train: %d" % train.count())

train_cols = train.columns
train_cols.remove('MachineIdentifier')
train_cols.remove('HasDetections')

# Convertimos el TRAIN en un VECTOR para poder pasarle el RF
print('Conversion de datos a VectorAssembler')
assembler_features = VectorAssembler(inputCols=train_cols, outputCol='features')
train_data = assembler_features.transform(train)

numFolds = 3

print('Creamos modelo')
rf = RandomForestClassifier(labelCol="HasDetections", featuresCol="features")
evaluator = BinaryClassificationEvaluator(rawPredictionCol="features",
                                          labelCol="HasDetections",
                                          metricName="areaUnderROC")

pipeline = Pipeline(stages=[rf])
paramGrid = ParamGridBuilder()\
    .addGrid(rf.numTrees, [3, 10])\
    .build()
# .addGrid(...)  # Add other parameters

print('Creamos cross-validador con {} folds'.format(numFolds))
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=numFolds)

print('Entrenando modelo...')
model = crossval.fit(train_data)
print('Fin del train')
print('Guardando el modelo...')
try:
    model.bestModel.save('RandomForest_0')
except:
    print('No se pudo guardar el modelo')

print(model.bestModel.summary)
print(model.bestModel.summary())

rfModel = model.stages[2]
print(rfModel)  # summary only

