from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("FindTable").getOrCreate()

# Databricks display function - fallback to show() for local execution
try:
    from databricks.sdk.runtime import display
except ImportError:
    display = lambda df: df.show()

dbs = [row['databaseName'] for row in spark.sql("SHOW DATABASES").collect()]
result = []
for db in dbs:
    tables = spark.sql(f"SHOW TABLES IN {db}").collect()
    for tbl in tables:
        if 'mongo_iceberg_applications' in tbl['tableName']:
            result.append({'database': db, 'table': tbl['tableName']})
display(spark.createDataFrame(result))