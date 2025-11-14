import os
import xml.etree.ElementTree as ET

# Input + output folders
xml_folder = "xml_files"   # folder where XML files are stored
output_folder = "labels"   # folder where YOLO labels will be saved
os.makedirs(output_folder, exist_ok=True)

# Define classes
classes = ["potholes"]   # your dataset classes

def convert_voc_to_yolo(size, box):
    """
    Convert VOC bbox to YOLO format
    size: (width, height)
    box: (xmin, ymin, xmax, ymax)
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    label_lines = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)

        xmlbox = obj.find("bndbox")
        xmin = int(xmlbox.find("xmin").text)
        ymin = int(xmlbox.find("ymin").text)
        xmax = int(xmlbox.find("xmax").text)
        ymax = int(xmlbox.find("ymax").text)

        # Convert to YOLO
        x, y, w, h = convert_voc_to_yolo((img_w, img_h), (xmin, ymin, xmax, ymax))
        label_lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    # Save to .txt (same filename as image)
    txt_filename = os.path.splitext(xml_file)[0] + ".txt"
    with open(os.path.join(output_folder, txt_filename), "w") as f:
        f.write("\n".join(label_lines))

print(f"✅ Conversion complete! YOLO labels are saved in '{output_folder}'")
