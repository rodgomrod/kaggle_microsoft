# PARA HACERLO ANDAR: pyspark --packages Azure:mmlspark:0.15   |    en consola

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
from mmlspark import LightGBMClassifier
from mmlspark.ComputeModelStatistics import ComputeModelStatistics

spark = SparkSession.builder.appName('LGBM_trainer').getOrCreate()

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

train_2 = train.limit(10000)
train_data = assembler_features.transform(train_2)
train_data = train_data.select('features', 'HasDetections')\
    .withColumnRenamed('HasDetections', 'label')

trainDF, testDF = train_data.randomSplit([0.8, 0.2], seed=24)

model = LightGBMClassifier(learningRate=0.1,
                           numIterations=30,
                           numLeaves=20,
                           objective="binary",
                           maxDepth=10
                           )

model_trained = model.fit(trainDF)

prediction = model_trained.transform(testDF)

metrics = ComputeModelStatistics().transform(prediction)
metrics.select('accuracy').show()

# >>> d = [{'name': 'Alice', 'age': 1}]
# >>> spark.createDataFrame(d).collect()


# FEATURES IMPORTANCES
# for i, j in zip(model.getFeatureImportances(), train_cols):
#     if i > 0:
#         print(i,j)
