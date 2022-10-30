import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lower, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import StructType as R, StructField as Fld, \
     DoubleType as Dbl, LongType as Long, StringType as Str, \
     IntegerType as Int, DecimalType as Dec, DateType as Date, \
     TimestampType as Stamp


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    spark.conf.set("mapreduce.fileoutputcommitter.algorithm.version", "2")
    return spark


def get_song_schema():
    """
    Creates a schema for song data.
    
    :return: schema
    """
    song_schema = R([
        Fld("num_songs", Int()),
        Fld("artist_id", Str()),
        Fld("artist_latitude", Dec()),
        Fld("artist_longitude", Dec()),
        Fld("artist_location", Str()),
        Fld("artist_name", Str()),
        Fld("song_id", Str()),
        Fld("title", Str()),
        Fld("duration", Dbl()),
        Fld("year", Long())
    ])
    return song_schema


def process_song_data(spark, input_data, output_data):
    
    # get filepath to song data file
    # song_data = input_data + "song_data/*/*/*/*.json"
    song_data = input_data

    # read song data file
    song_df = spark.read.json(song_data,schema=get_song_schema())
    
    # Convert title and artist_name to lowercase
    columns = ['title','artist_name']
    for colName in columns:
        song_df = song_df.withColumn(colName, lower(col(colName)))

    # extract columns to create songs table
    songs_table = song_df.select("song_id",
                        "title",
                        "artist_id",
                        "year",
                        "duration").dropDuplicates(["song_id"]) 
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(output_data + "songs_table.parquet",
                          partitionBy = ["year", "artist_id"],
                          mode = "overwrite") 

    # extract columns to create artists table
    artists_table = song_df.select("artist_id",
                          "artist_name",
                          "artist_location",
                          "artist_latitude",
                          "artist_longitude").dropDuplicates(["artist_id"])
    
    # write artists table to parquet files
    artists_table.write.parquet(output_data + "artists_table.parquet", mode = "overwrite")


def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    #log_data = input_data + "log-data/*/*/*.json"
    log_data = input_data
    
    log_schema = R([
        Fld("artist", Str()),
        Fld("auth", Str()),
        Fld("firstName", Str()),
        Fld("gender", Str()),
        Fld("itemInSession", Str()),
        Fld("lastName", Str()),
        Fld("length", Dbl()),
        Fld("level", Str()),
        Fld("location", Str()),
        Fld("method", Str()),
        Fld("page", Str()),
        Fld("registration", Dbl()),
        Fld("sessionId", Str()),
        Fld("song", Str()),
        Fld("status", Str()),
        Fld("ts", Long()),
        Fld("userAgent", Str()),
        Fld("userId", Str())
    ])

    # read log data file
    log_df = spark.read.json(log_data, schema=log_schema)
    
    # filter by actions for song plays
    log_df = log_df.filter(log_df.page == "NextSong")
    
    # Convert song and artist to lowercase
    columns = ['song','artist']
    for colName in columns:
        log_df = log_df.withColumn(colName, lower(col(colName)))

    # extract columns for users table    
    users_table = log_df.selectExpr("userId as user_id",
                            "firstName as first_name",
                            "lastName as last_name",
                            "gender",
                            "level").dropDuplicates(["user_id"]) 
    
    # write users table to parquet files
    users_table.write.parquet("output_data/" + "users_table.parquet",mode = "overwrite")

    # create timestamp column from original timestamp column
    #log_df = log_df.withColumn('timestamp',( (log_df.ts.cast('float')/1000).cast("timestamp")) )
    get_timestamp = udf(lambda x: datetime.fromtimestamp((x / 1000)), Stamp())
    log_df = log_df.withColumn("timestamp", get_timestamp(col("ts")))
    
    # extract columns to create time table
    time_table = log_df.selectExpr("timestamp as start_time",
                           "hour(timestamp) as hour",
                           "dayofmonth(timestamp) as day",
                           "weekofyear(timestamp) as week",
                           "month(timestamp) as month",
                           "year(timestamp) as year",
                           "dayofweek(timestamp) as weekday"
                           ).dropDuplicates(["start_time"])
    
    # write time table to parquet files partitioned by year and month
    time_table.write.parquet("output_data/"  + "time_table.parquet",
                         partitionBy = ["year", "month", "week"],
                         mode = "overwrite")

    # read in song data to use for songplays table
    song_df = spark.read.json("song-data/song_data/*/*/*/",schema=get_song_schema())

    song_log_joined_df = log_df.join(song_df, (log_df.song == song_df.title) & (log_df.artist == song_df.artist_name) & (log_df.length == song_df.duration), how='inner')
    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = songplays_table = song_log_joined_df.distinct() \
                    .selectExpr("userId as user_id", "timestamp", "song_id", "artist_id", "level", "sessionId", "location", "userAgent", \
                                "year(timestamp) as year","month(timestamp) as month","weekofyear(timestamp) as week") \
                    .withColumn("songplay_id", monotonically_increasing_id()) \
                    .withColumnRenamed("userId","user_id")        \
                    .withColumnRenamed("timestamp","start_time")  \
                    .withColumnRenamed("sessionId","session_id")  \
                    .withColumnRenamed("userAgent", "user_agent")   

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.parquet("output_data/" + "songplays_table.parquet",
                              partitionBy=["year", "month"],
                              mode="overwrite")


def main():
    spark = create_spark_session()
    # input_data = "s3://udacity-dend/song_data"
    output_data = "output_data/"
    
    input_data = "song-data/song_data/*/*/*/"
    process_song_data(spark, input_data, output_data)    
    
    input_data = "log-data/"
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
