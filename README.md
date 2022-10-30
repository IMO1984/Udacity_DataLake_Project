# Project: Data Lake with Spark

## Introduction

<p>A music streaming startup, Sparkify, has grown their user base and song database even more and want to move their data warehouse to a data lake. Their data resides in S3, in a directory of JSON logs on user activity on the app, as well as a directory with JSON metadata on the songs in their app.</p>

<p>Sparkify want to build an ETL pipeline that extracts their data from S3, processes them using Spark, and loads the data back into S3 as a set of dimensional tables. This will allow their analytics team to continue finding insights in what songs their users are listening to.</p>


## Project Datasets

### Song Dataset

The first dataset is a subset of real data from the [Million Song Dataset](http://millionsongdataset.com/).
Each file is in JSON format and contains metadata about a song and the artist of that song. 

The files are partitioned by the first three letters of each song's track ID. For example, here are filepaths to two files in this dataset.
>**s3://udacity-dend/song_data/A/A/A/TRAAAAW128F429D538.json**<br>

Below is an example of what a single song file, **TRAABJL12903CDCF1A.json**, looks like.<br>
```
{
"num_songs": 1, 
"artist_id": "ARD7TVE1187B99BFB1", 
"artist_latitude": null, 
"artist_longitude": null, 
"artist_location": "California - LA", 
"artist_name": "Casual", 
"song_id": "SOMZWCG12A8C13C480", 
"title": "I Didn't Mean To", 
"duration": 218.93179, 
"year": 0} 

```
### Log Dataset

The second dataset consists of log files in JSON format generated by this [event simulator](https://github.com/Interana/eventsim) based on the songs in the dataset above. These simulate activity logs from a music streaming app based on specified configurations.

The log files in the dataset are partitioned by year and month. For example, here are filepaths to two files in this dataset.

>**s3://udacity-dend/log_data/2018/11/2018-11-12-events.json**<br>


Below is an example of what the data in a log file, **2018-11-12-events.json**, looks like.
![Log data example!](./log-data.png "Log data example")

## Schema for Song Play Analysis
Using the song and log datasets, below mentioned star schema optimized for queries on song play analysis need to be created. 
This includes the following tables.
**Fact Table
1. songplays - records in log data associated with song plays i.e. records with page NextSong
> songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent

Dimension Tables
2. users - users in the app
> user_id, first_name, last_name, gender, level
3. songs - songs in music database
> song_id, title, artist_id, year, duration
4. artists - artists in music database
> artist_id, name, location, lattitude, longitude
5. time - timestamps of records in songplays broken down into specific units
> start_time, hour, day, week, month, year, weekday


![Star Schema for Song Play Analysis!](./song_play_analysis_with_star_schema.png "Star Schema for Song Play Analysis")

## <br>Data Lake to store extracted dimentional tables
>**"s3a://udacity-de-sparkify-data-lake/artists" <br>
>"s3a://udacity-de-sparkify-data-lake/songs" <br>
>"s3a://udacity-de-sparkify-data-lake/time" <br>
>"s3a://udacity-de-sparkify-data-lake/users" <br>
>"s3a://udacity-de-sparkify-data-lake/songplays"**<br>


## <br>Project Files

In addition to the data files, the project workspace includes 5 files:

**1. dl.cfg**                    Contains the Secret Key for ASW access<br>
**2. create_bucket.py**          Create bucket in AWS S3 to store the extracted dimentional tables.<br>
**3. etl.py**                    Loading song data and log data from S3 to Spark, transforms data into a set of dimensional tables, then save the table back to S3 <br>
**4. etl.ipynb**                 Used to design ETL pipelines <br>
**5. extractFiles		 Used to unzip local files.
**6. README.md**                 Provides project info<br>

## Configuration

Remember to set key and secret in **dl.cfg** before run **etl.py**<br>

[AWS]<br>
key = <br>
secret = <br>

## Build ETL Pipeline

**etl.py** will process the entire datasets.


## Instruction

1. Set **key** and **secrect** in **dl.cfg** file <br><br>

2. Run **create_bucket.py**<br>
    **python create_bucket.py** <br> <br>
    
3. Use following command to start ETL process <br>
    **python etl.py** <br> <br>

4. Need spark(Localy) or can be executed directly on AWS EMR. Please be mindful about the required imports.

    


