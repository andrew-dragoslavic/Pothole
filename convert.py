import os
import xml.etree.ElementTree as ET
from shutil import copyfile

xml_dir = "potholes/annotations"
images_dir = "potholes/images"
output_images_dir = "potholes/output/train/images"
output_annotations_dir = "potholes/output/train/labels"

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_annotations_dir, exist_ok=True)

class_id = 0

def convert_bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h):
    x_center = (x_max + x_min) / 2.0
    y_center = (y_max + y_min) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h

    return x_center, y_center, w, h

for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text
    image_path = os.path.join(images_dir, filename)
    if not os.path.isfile(image_path):
        continue

    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)

    yolo_lines = []

    for obj in root.findall('object'):
        bnd_box = obj.find('bndbox')
        x_min = float(bnd_box.find('xmin').text)
        y_min = float(bnd_box.find('ymin').text)
        x_max = float(bnd_box.find('xmax').text)
        y_max = float(bnd_box.find('ymax').text)

        x_center, y_center, w, h = convert_bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)
        yolo_lines.append(f"{class_id} {x_center} {y_center} {w} {h}")

    base_name = os.path.splitext(filename)[0]
    txt_output_path = os.path.join(output_annotations_dir, base_name + ".txt")
    with open(txt_output_path, "w") as txt_file:
        txt_file.write("\n".join(yolo_lines))

    copyfile(image_path, os.path.join(output_images_dir, filename))
