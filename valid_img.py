import os
import shutil

# Paths
images_dir = r"C:\Users\dions\Downloads\data\data\train\images"   # path to your images folder
labels_dir = r"C:\Users\dions\Downloads\data\data\train\labels"# path to your labels folder
output_images = r"C:\Users\dions\Downloads\cleaned\images"
output_labels = r"C:\Users\dions\Downloads\cleaned\labels"

# Create output dirs
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# Get file lists
image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.lower().endswith('.txt')}

# Find common base names
common = image_files & label_files
print(f"✅ Found {len(common)} matched pairs")
print(f"❌ {len(image_files - label_files)} images without labels")
print(f"❌ {len(label_files - image_files)} labels without images")

# Copy only matched pairs
for base in common:
    img_path = os.path.join(images_dir, base + ".jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(images_dir, base + ".png")  # fallback if PNG

    lbl_path = os.path.join(labels_dir, base + ".txt")

    if os.path.exists(img_path) and os.path.exists(lbl_path):
        shutil.copy(img_path, output_images)
        shutil.copy(lbl_path, output_labels)

print("✅ Clean dataset created in 'cleaned/' folder.")
# import os
# import shutil
# import random
#
# # Paths
# image_folder = r"C:\Users\dions\Downloads\cleaned\images"
# text_folder = r"C:\Users\dions\Downloads\cleaned\labels"
#
# print(os.listdir(image_folder))
#
# output_base = r"C:\Users\dions\Downloads\data"
# train_img = os.path.join(output_base, 'train', 'images')
# train_txt = os.path.join(output_base, 'train', 'labels')
# valid_img = os.path.join(output_base, 'valid', 'images')
# valid_txt = os.path.join(output_base, 'valid', 'labels')
#
# # Create directories
# for folder in [train_img, train_txt, valid_img, valid_txt]:
#     os.makedirs(folder, exist_ok=True)
#
# # List and shuffle image files
# image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
# random.shuffle(image_files)
#
# # Split 80/20
# split_index = int(0.8 * len(image_files))
# train_files = image_files[:split_index]
# valid_files = image_files[split_index:]
#
# def move_files(image_list, dest_img_folder, dest_txt_folder):
#     for img_file in image_list:
#         base_name, _ = os.path.splitext(img_file)
#         txt_file = base_name + '.txt'
#
#         # Copy image
#         src_img = os.path.join(image_folder, img_file)
#         dst_img = os.path.join(dest_img_folder, img_file)
#         if os.path.exists(src_img):
#             shutil.copy(src_img, dst_img)
#
#         # Copy corresponding text file (create empty if missing)
#         src_txt = os.path.join(text_folder, txt_file)
#         dst_txt = os.path.join(dest_txt_folder, txt_file)
#         if os.path.exists(src_txt):
#             shutil.copy(src_txt, dst_txt)
#         else:
#             open(dst_txt, 'w').close()
#
# # Move (copy) files
# move_files(train_files, train_img, train_txt)
# move_files(valid_files, valid_img, valid_txt)
#
# print("✅ Files successfully copied and organized under 'data/train' and 'data/valid'")
