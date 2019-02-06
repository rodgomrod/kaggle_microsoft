

def label_encoding_col(df,column):
    output_name = column + "_Index"
    indexer = StringIndexer(inputCol=column, outputCol=output_name)
    df = indexer.fit(df).transform(df)
    return df

def count_encoder_col(df,column):
    #Encoder por la frecuencia de aparición de valores

    output_name = column + "_Frecuency"
    frequency= df.groupBy(column).count().withColumnRenamed('count',output_name)
    df = df.join(frequency, column, 'left')
    return df
