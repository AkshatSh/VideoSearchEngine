import database_utils

def test_connection():
    # database_utils.upload_new_summary(
    #     "test video name", 
    #     "there is nothing here just to test proper db", 
    #     "fake_url.com"
    # )
    print(database_utils.get_all_data())

def test_update():
    database_utils.update_summary(
        update_id="5affc6e194dba95204adceaf",
        video_summary="new video summary updated version"
    )

def test_get_all_id():
    print(database_utils.get_all_ids())

def main():
    # test_update()
    test_connection()
    test_get_all_id()

if __name__ == '__main__':
    main()