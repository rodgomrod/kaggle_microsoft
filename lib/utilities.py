

def label_encoding_col(df,column):
    output_name = column + "Index"
    indexer = StringIndexer(inputCol=column, outputCol=output_name)
    df = indexer.fit(df).transform(df)
    return df