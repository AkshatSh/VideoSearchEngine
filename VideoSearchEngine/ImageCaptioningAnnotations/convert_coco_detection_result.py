import os
import ast
with open("data/coco_detection_result") as coco_detection_result_file:
    line = coco_detection_result_file.readline()
    result_dict = ast.literal_eval(line)
    file_path = result_dict['file_path']
    
    print(result_dict)
