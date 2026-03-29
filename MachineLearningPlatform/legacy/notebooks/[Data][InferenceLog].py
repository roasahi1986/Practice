from pyspark.sql import SparkSession, types as T, functions as F
from pyspark.sql.protobuf.functions import from_protobuf

spark = SparkSession.builder.getOrCreate()

self_serv_path = "coba2/data_multi_az/ex-machine_learning-logs"
INPUT_BUCKET = "exchange-logs"
INPUT_SELF_SERV_LOG_PATH=f"s3://{INPUT_BUCKET}/{self_serv_path}"

self_serv_paths = []
for hr in range(1):
    self_serv_path = [INPUT_SELF_SERV_LOG_PATH,
                f'dt=2025-11-09',
                f'hr={hr:02}']
    self_serv_path = '/'.join(self_serv_path)
    self_serv_paths.append(self_serv_path)

self_serv_schema = T.StructType([
    T.StructField("dt", T.StringType()),
    T.StructField("hr", T.StringType()),
    T.StructField("project_name", T.StringType()),
    T.StructField("experiment_id", T.LongType()),
    T.StructField("event_id", T.StringType()),
    T.StructField("timestamp", T.TimestampType()),
    T.StructField("traffic_allocation", T.DoubleType()),
    T.StructField("downsampling_rate", T.DoubleType()),
    T.StructField("features", T.StringType()),
    T.StructField("predictions", T.StringType()),
    T.StructField("tags", T.StringType()),
])

self_serv_view_name = "luxu_tmp_view"
(
    spark
        .read
        .schema(self_serv_schema)
        .option("mergeSchema", "True")
        .option("basePath", INPUT_SELF_SERV_LOG_PATH)
        .parquet(*self_serv_paths)
        .createOrReplaceTempView(self_serv_view_name)
)

DESCRIPTOR_PATH="/Workspace/Users/luxu/Dependency/message.desc"
query = f"""
SELECT experiment_id, features, predictions FROM {self_serv_view_name}
"""
df = spark.sql(query)
df = df.withColumn(
    'decoded_features', F.unbase64(df.features)
).withColumn(
    'decoded_predictions', F.unbase64(df.predictions)
)
df = df.withColumn(
    'deserialized_features',
    from_protobuf(
        df.decoded_features, 'MLFeatures', DESCRIPTOR_PATH
    )
)
df = df.withColumn(
    'deserialized_predictions',
    from_protobuf(
        df.decoded_predictions, 'MLPredictions', DESCRIPTOR_PATH
    )
)