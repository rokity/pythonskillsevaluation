import sys
import cv2
import os
import xml.etree.ElementTree as ET
import json
import numpy as np


SIZE=(800,450)
OUTPUT_JSON_FILENAME='output.json'

#Read Images from input directory and resize to 800x450 and save to output directory
# @param image_dir: input directory
# @param output_dir: output directory
def read_images_resize_save_output_dir(path, output_dir):
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image = cv2.imread(os.path.join(path, filename))
            if image.shape[0] > SIZE[0] and image.shape[1] > SIZE[1]:
                image = cv2.resize(image, SIZE)
            cv2.imwrite(os.path.join(output_dir, filename), image)

# Parse Annotations from Objects on the XML file
# @param objects: list of objects from xml file
# @param width: width of image
# @param height: height of image
# @param image_id: image id
# @return: list of annotations in coco format
def parse_objects_info(objects,width,height,image_id):
    coco_annotations=[]
    if height > SIZE[1] and width > SIZE[0]:
        y_scale=SIZE[1]/height
        x_scale=SIZE[0]/width
    else:
        y_scale=1
        x_scale=1
        
    for i in range(0,len(objects)):
        class_name = objects[i].find('name').text
        bndbox = objects[i].find('bndbox')
        xmin = int(np.round(int(bndbox.find('xmin').text)*x_scale))
        ymin = int(np.round(int(bndbox.find('ymin').text)*y_scale))
        xmax = int(np.round(int(bndbox.find('xmax').text)*x_scale))
        ymax = int(np.round(int(bndbox.find('ymax').text)*y_scale))

        coco_annotations.append({'image_id': image_id,
                                 'category_id': class_name,
                                 'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                                 'id': f"{image_id}_{i}"})
    return coco_annotations

# Read XML file formatted in VOC , extract image information and objects annotations
# @param xml_path: path to xml file
# @return coco_annotations,image_object: list of annotations in coco format and image information
def read_xml_voc_format_and_annotate_coco_format(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    filename = root.find('filename').text
    image_id = filename.split('.')[0]
    image_object={'width': width,'height': height,'file_name': filename,'id': image_id}    
    objects = root.findall('object')    
    coco_annotations = parse_objects_info(objects,width,height,image_id)    
    return coco_annotations,image_object

# Parse XML files and save to output directory
# @param xml_dir: path to xml files
# @param output_dir: path to output directory
# @param output_json_file: name of output json file
def parse_xml_data(xml_input_dir,output_dir,output_json_file):
    annotations=[]
    image_objects=[]
    for filename in os.listdir(xml_input_dir):
        _annotations,_image_object=read_xml_voc_format_and_annotate_coco_format(os.path.join(xml_input_dir,filename))
        image_objects.append(_image_object)
        annotations=annotations+_annotations
    #Dyanmically classification of annotations
    _categories=[]
    _category_objects=[]
    _id=0
    for i in range(0,len(annotations)):
        if annotations[i]['category_id'] not in _categories:
            _categories.append(annotations[i]['category_id'])            
            _category_objects.append({'id':_id,'name':annotations[i]['category_id'],"supercategory":None})
            annotations[i]['category_id']=_id
            _id+=1
    with open(os.path.join(output_dir,output_json_file), 'w') as fp:
        json.dump({"categories":_categories,"images":image_objects,"annotations":annotations}, fp)
        
def main():
    if len(sys.argv) != 7:
        print("Usage: python app.py --imagedir <path_to_images> --xmdir <path_to_xml> --outputdir <path_to_output>")
        sys.exit(1)
    image_dir = sys.argv[2]
    xml_dir = sys.argv[4]
    output_dir = sys.argv[6]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    read_images_resize_save_output_dir(image_dir, output_dir)
    parse_xml_data(xml_dir,output_dir,OUTPUT_JSON_FILENAME)


if __name__ == '__main__':
    main()