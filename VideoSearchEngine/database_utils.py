import pymongo
from mongoengine import (
    Document,
    StringField,
)
import config
from bson.objectid import ObjectId


'''
This util package contains all the information to interact with the database storing the videos

structure of the database is: 
{
    "name" : "video name",
    "summary": "video summary",
    "url": "video url"
}


class VideoEntry(Document):
    name = StringField(required=True, max_length=200)
    summary = StringField(required=True)
    url = StringField(required=True)
'''

client = pymongo.MongoClient(config.VIDEO_DATABASE_URL)
db = client[config.VIDEO_DATABASE_NAME] # videosearchdatabase
collection = db[config.VIDEO_COLLECTION_NAME]\


def get_entry(entry_id):
    return collection.find_one({"_id" : ObjectId(entry_id)})

def get_summary(entry_id):
    return get_entry(entry_id)['summary']

def get_url(entry_id):
    return get_entry(entry_id)['url']

def get_video_name(entry_id):
    return get_entry(entry_id)['name']

def get_id_from_name(name):
    return collection.find({"name" : name})[0]['_id']

def get_all_ids():
    return [str(entry['_id']) for entry in get_all_data()]

def get_all_data():
    result = []
    for entry in collection.find():
        result.append(entry)
    return result

def get_all_summaries():
    result = []
    for entry in collection.find():
        result.append(entry['summary'])
    return result

def upload_new_summary(video_name, video_summary, video_url):
    return collection.insert_one(
        {
            "name" : video_name,
            "summary": video_summary,
            "url": video_url
        }
    )

def update_summary(
        update_id, 
        video_name=None, 
        video_summary=None, 
        video_url=None
    ):

    update_data = {}

    if video_name is not None:
        update_data["name"] = video_name
    
    if video_summary is not None:
        update_data["summary"] = video_summary
    
    if video_url is not None:
        update_data["url"] = video_url
    
    result = collection.update_one(
        {"_id": ObjectId(update_id)},
        {"$set": update_data}
    )

    return result