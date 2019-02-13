# PARA HACERLO ANDAR: pyspark --packages Azure:mmlspark:0.15   |    en consola

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.types import IntegerType, StringType, FloatType, StructType, StructField, LongType

schema_train = StructType([StructField("MachineIdentifier", StringType(), True),
                            StructField("IsBeta", IntegerType(), True),
                            StructField("RtpStateBitfield", IntegerType(), True),
                            StructField("IsSxsPassiveMode", IntegerType(), True),
                            StructField("DefaultBrowsersIdentifier", IntegerType(), True),
                            StructField("AVProductStatesIdentifier", IntegerType(), True),
                            StructField("AVProductsInstalled", IntegerType(), True),
                            StructField("AVProductsEnabled", IntegerType(), True),
                            StructField("HasTpm", IntegerType(), True),
                            StructField("CountryIdentifier", IntegerType(), True),
                            StructField("CityIdentifier", IntegerType(), True),
                            StructField("OrganizationIdentifier", IntegerType(), True),
                            StructField("GeoNameIdentifier", IntegerType(), True),
                            StructField("LocaleEnglishNameIdentifier", IntegerType(), True),
                            StructField("OsBuild", IntegerType(), True),
                            StructField("OsSuite", IntegerType(), True),
                            StructField("IsProtected", IntegerType(), True),
                            StructField("AutoSampleOptIn", IntegerType(), True),
                            StructField("SMode", IntegerType(), True),
                            StructField("IeVerIdentifier", IntegerType(), True),
                            StructField("Firewall", IntegerType(), True),
                            StructField("UacLuaenable", IntegerType(), True),
                            StructField("Census_OEMNameIdentifier", IntegerType(), True),
                            StructField("Census_OEMModelIdentifier", IntegerType(), True),
                            StructField("Census_ProcessorCoreCount", IntegerType(), True),
                            StructField("Census_ProcessorManufacturerIdentifier", IntegerType(), True),
                            StructField("Census_ProcessorModelIdentifier", IntegerType(), True),
                            StructField("Census_PrimaryDiskTotalCapacity", LongType(), True),
                            StructField("Census_SystemVolumeTotalCapacity", IntegerType(), True),
                            StructField("Census_HasOpticalDiskDrive", IntegerType(), True),
                            StructField("Census_TotalPhysicalRAM", IntegerType(), True),
                            StructField("Census_InternalPrimaryDiagonalDisplaySizeInInches", FloatType(), True),
                            StructField("Census_InternalPrimaryDisplayResolutionHorizontal", IntegerType(), True),
                            StructField("Census_InternalPrimaryDisplayResolutionVertical", IntegerType(), True),
                            StructField("Census_InternalBatteryNumberOfCharges", LongType(), True),
                            StructField("Census_OSBuildNumber", IntegerType(), True),
                            StructField("Census_OSBuildRevision", IntegerType(), True),
                            StructField("Census_OSInstallLanguageIdentifier", IntegerType(), True),
                            StructField("Census_OSUILocaleIdentifier", IntegerType(), True),
                            StructField("Census_IsPortableOperatingSystem", IntegerType(), True),
                            StructField("Census_IsFlightingInternal", IntegerType(), True),
                            StructField("Census_IsFlightsDisabled", IntegerType(), True),
                            StructField("Census_ThresholdOptIn", IntegerType(), True),
                            StructField("Census_FirmwareManufacturerIdentifier", IntegerType(), True),
                            StructField("Census_FirmwareVersionIdentifier", IntegerType(), True),
                            StructField("Census_IsSecureBootEnabled", IntegerType(), True),
                            StructField("Census_IsWIMBootEnabled", IntegerType(), True),
                            StructField("Census_IsVirtualDevice", IntegerType(), True),
                            StructField("Census_IsTouchEnabled", IntegerType(), True),
                            StructField("Census_IsPenCapable", IntegerType(), True),
                            StructField("Census_IsAlwaysOnAlwaysConnectedCapable", IntegerType(), True),
                            StructField("Wdft_IsGamer", IntegerType(), True),
                            StructField("Wdft_RegionIdentifier", IntegerType(), True),
                            StructField("HasDetections", IntegerType(), True),
                            StructField("Census_InternalBatteryType_informed", IntegerType(), True),
                            StructField("ProductName_index", IntegerType(), True),
                            StructField("Platform_index", IntegerType(), True),
                            StructField("Processor_index", IntegerType(), True),
                            StructField("OsPlatformSubRelease_index", IntegerType(), True),
                            StructField("SkuEdition_index", IntegerType(), True),
                            StructField("PuaMode_index", IntegerType(), True),
                            StructField("SmartScreen_index", IntegerType(), True),
                            StructField("Census_MDC2FormFactor_index", IntegerType(), True),
                            StructField("Census_DeviceFamily_index", IntegerType(), True),
                            StructField("Census_ProcessorClass_index", IntegerType(), True),
                            StructField("Census_PrimaryDiskTypeName_index", IntegerType(), True),
                            StructField("Census_ChassisTypeName_index", IntegerType(), True),
                            StructField("Census_PowerPlatformRoleName_index", IntegerType(), True),
                            StructField("Census_InternalBatteryType_index", IntegerType(), True),
                            StructField("Census_OSArchitecture_index", IntegerType(), True),
                            StructField("Census_OSBranch_index", IntegerType(), True),
                            StructField("Census_OSEdition_index", IntegerType(), True),
                            StructField("Census_OSSkuName_index", IntegerType(), True),
                            StructField("Census_OSInstallTypeName_index", IntegerType(), True),
                            StructField("Census_OSWUAutoUpdateOptionsName_index", IntegerType(), True),
                            StructField("Census_GenuineStateName_index", IntegerType(), True),
                            StructField("Census_ActivationChannel_index", IntegerType(), True),
                            StructField("Census_FlightRing_index", IntegerType(), True),
                            StructField("Census_OSVersion_0", IntegerType(), True),
                            StructField("Census_OSVersion_1", IntegerType(), True),
                            StructField("Census_OSVersion_2", IntegerType(), True),
                            StructField("Census_OSVersion_3", IntegerType(), True),
                            StructField("EngineVersion_2", IntegerType(), True),
                            StructField("EngineVersion_3", IntegerType(), True),
                            StructField("AppVersion_1", IntegerType(), True),
                            StructField("AppVersion_2", IntegerType(), True),
                            StructField("AppVersion_3", IntegerType(), True),
                            StructField("AvSigVersion_0", IntegerType(), True),
                            StructField("AvSigVersion_1", IntegerType(), True),
                            StructField("AvSigVersion_2", IntegerType(), True),
                            StructField("OsVer_0", IntegerType(), True),
                            StructField("OsVer_1", IntegerType(), True),
                            StructField("OsVer_2", IntegerType(), True),
                            StructField("OsVer_3", IntegerType(), True),
                            StructField("OsBuildLab_0", IntegerType(), True),
                            StructField("OsBuildLab_1", IntegerType(), True),
                            StructField("OsBuildLab_4_0", IntegerType(), True),
                            StructField("OsBuildLab_4_1", IntegerType(), True),
                            StructField("OsBuildLab_2_index", IntegerType(), True),
                            StructField("OsBuildLab_3_index", IntegerType(), True)])


spark = SparkSession.builder.appName('LGBM_trainer') \
    .config("spark.jars.packages", "Azure:mmlspark:0.15") \
    .getOrCreate()

from mmlspark import LightGBMClassifier
from mmlspark.ComputeModelStatistics import ComputeModelStatistics
import mmlspark

path_train = '../../data/train_final_0'
print('Carga del TRAIN', path_train)
train = spark.read.csv(path=path_train, header=True, sep=',', inferSchema=True)

# train.persist()
# print("Numero de casos en el train: %d" % train.count())

ignore_c = ['MachineIdentifier', 'HasDetections']
train_cols = [c for c in train.columns if c not in ignore_c]

# Convertimos el TRAIN en un VECTOR para poder pasarle el RF
print('Conversion de datos a VectorAssembler')
assembler_features = VectorAssembler(inputCols=train_cols, outputCol='features')

train_2 = train.limit(1000000)
train_data = assembler_features.transform(train_2)
train_data = train_data.select('features', 'HasDetections')\
    .withColumnRenamed('HasDetections', 'label')

train_data.persist()
train_data.first()

trainDF, testDF = train_data.randomSplit([0.8, 0.2], seed=24)

model = LightGBMClassifier(learningRate=0.05,
                           numIterations=60,
                           numLeaves=30,
                           objective="binary",
                           maxDepth=13,
                           earlyStoppingRound=7,
                           featureFraction=0.7
                           )

model_trained = model.fit(trainDF)

prediction = model_trained.transform(testDF)

metrics = ComputeModelStatistics(evaluationMetric='AUC',
                                 labelCol='label',
                                 scoredLabelsCol=None,
                                 scoresCol=None).transform(prediction)
metrics.select('accuracy').show()

# >>> d = [{'name': 'Alice', 'age': 1}]
# >>> spark.createDataFrame(d).collect()


# FEATURES IMPORTANCES
# for i, j in zip(model.getFeatureImportances(), train_cols):
#     if i > 0:
#         print(i,j)
