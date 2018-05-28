import json
import os
os.system('wget http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2014.zip')
os.system('unzip image_info_test2014.zip')

train_val_info = json.load(open('data/coco_raw.json', 'r'))

filepath2idx = {entry['file_path']:i for i, entry in enumerate(train_val_info)}

# for i, entry in enumerate(train_val_info):
#     filepath2idx[entry['file_path']] = i

test_info = json.load(open('data/annotations/image_info_test2014.json', 'r'))['images']
testfilename2id = {info['file_name']:info['id'] for info in test_info}

detection_result = list(map(json.loads, open('data/coco_detection_result', 'r').readlines()))

# convert absolute path to relative path
for detection in detection_result:
    abs_path = detection['file_path']
    # rfind returns -1 if not found
    idx = max(abs_path.rfind('train2014/'), abs_path.rfind('val2014/'), abs_path.rfind('test2014/'))
    detection['file_path'] = detection['file_path'][idx:]

# add captions and ids to detection result
for i, detection in enumerate(detection_result):
    filepath = detection['file_path']
    if 'val2014/' in filepath or 'train2014/' in filepath:
        detection['captions'] = train_val_info[filepath2idx[filepath]]['captions']
        detection['id'] = train_val_info[filepath2idx[filepath]]['id']
    elif 'test2014/' in filepath:
        detection['captions'] = ['dummy caption for test images' for i in range(5)]
        filename = filepath[len('test2014/'):]
        detection['id'] = testfilename2id[filename]
    else:
        raise ('filepath not matched')

# process categories
for detection in detection_result:
    detection['categories'] = list(set(detection['full_categories']))
    # 96 is the max encoder sequence length
    if len(detection['full_categories']) > 96:
        detection['full_categories'] = detection['full_categories'][:96]


# let item orders of resulting json file follows coco_raw.json
filepath2idx_detection = {detection['file_path']:i for i, detection in enumerate(detection_result)}
detection_result_reorder = [detection_result[filepath2idx_detection[info['file_path']]] for info in train_val_info]
for detection in detection_result:
    if 'test2014/' in detection['file_path']:
        detection_result_reorder.append(detection)

for i in range(len(train_val_info)):
    assert(train_val_info[i]['captions'] == detection_result_reorder[i]['captions'])
    assert(train_val_info[i]['file_path'] == detection_result_reorder[i]['file_path'])
    assert(train_val_info[i]['id'] == detection_result_reorder[i]['id'])


# write resulting json file
with open("coco_yolo_objname_location.json", 'w') as outfile:
    json.dump(detection_result_reorder, outfile)