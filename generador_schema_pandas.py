from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import numpy as np

spark = SparkSession.builder.appName('generador_schema').getOrCreate()
train_path = 'data/train_final_3/*.csv'
train = spark.read.csv(train_path, header=True, inferSchema=True)

train.persist()
print(train.count(), '\n')


cols = train.columns
cols.remove('MachineIdentifier')

print('{} columnas a procesar'.format(len(cols)))

for c in cols:
    maximo = train.select(c).agg(max(col(c))).collect()[0][0]
    if maximo > 2**7:
            if maximo > 2**15:
                    print("'{}': np.int32,".format(c))
            else:
                    print("'{}': np.int16,".format(c))
    else:
            print("'{}': np.int8,".format(c))



