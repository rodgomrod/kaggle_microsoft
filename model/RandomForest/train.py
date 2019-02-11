from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier

spark = SparkSession.builder.appName('RF_trainer').getOrCreate()

path_train = 'data/train_final_0'
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
# train_data.show(10)

kFolds = 3

print('Creamos modelo')
rf = RandomForestClassifier(labelCol="HasDetections", featuresCol="features")
evaluator = BinaryClassificationEvaluator(rawPredictionCol="features",
                                          labelCol="HasDetections",
                                          metricName="areaUnderROC")

pipeline = Pipeline(stages=[rf])
paramGrid = ParamGridBuilder()\
    .addGrid(rf.numTrees, [10])\
    .addGrid(rf.setSeed, [1])\
    .addGrid(rf.setMaxDepth, [7, 9])\
    .build()


# .addGrid(...)  # Add other parameters

print('Creamos cross-validador con {} folds'.format(kFolds))
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=kFolds)

print('Entrenando modelo...')
model = crossval.fit(train_data)
print('Fin del train')
print('Guardando el modelo...')
try:
    model.bestModel.save('RandomForest_0')
except:
    print('No se pudo guardar el modelo')

try:
    for i, j in zip(model.avgMetrics, paramGrid):
        print("Score {} con los parametros {}".format(i, j))
except:
    print("No se pudo hacer zip(model.avgMetrics, paramGrid)")

try:
    bestPipeline = model.bestModel
    bestLRModel = bestPipeline.stages[2]
    bestParams = bestLRModel.extractParamMap()
    print("Best pipeline", bestPipeline)
    print("Best model", bestLRModel)
    print("Best params", bestParams)
except:
    pass

try:
    print(model.bestModel.summary)
except:
    print("No se pudo hacer model.bestModel.summary")

try:
    rfModel = model.stages[2]
    print(rfModel)  # summary only
except:
    print("No se pudo hacer model.stages[2]")
