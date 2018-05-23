# all configuration options to run the video search engine
from database_settings import (
    VIDEO_DATABASE_USER,
    VIDEO_DATABASE_PASSWORD,
)


# ---- VIDEO DATABASE CONFIGURATION SECTION ---
VIDEO_DATABASE_NAME = "videosearchdatabase"
VIDEO_COLLECTION_NAME = "videos"
VIDEO_DATABASE_LOCATION = "ds229450.mlab.com:29450"

VIDEO_DATABASE_URL = "mongodb://{user}:{password}@{location}/{database}".format(
    user=VIDEO_DATABASE_USER,
    password=VIDEO_DATABASE_PASSWORD,
    location=VIDEO_DATABASE_LOCATION,
    database=VIDEO_DATABASE_NAME,
)

OBJECT_DETECTION_TYPE = "TINY_YOLO"