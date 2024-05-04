import shutil
from time import process_time
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.util import img_as_ubyte


def directory_creation(class_name, dir):
    train_path = os.path.join(dir, 'train', class_name)
    os.makedirs(train_path)
    val_path = os.path.join(dir, 'val', class_name)
    os.makedirs(val_path)
    test_path = os.path.join(dir, 'test', class_name)
    os.makedirs(test_path)


def create_partition(root_dir, currentCls, final_dir):
    src = os.path.join(root_dir, currentCls)
    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = (np.split(np.array(allFileNames), [int(len(allFileNames) * 0.8), int(len(allFileNames) * 0.9)]))
    train_FileNames = [os.path.join(src, name) for name in train_FileNames.tolist()]
    val_FileNames = [os.path.join(src, name) for name in val_FileNames.tolist()]
    test_FileNames = [os.path.join(src, name) for name in test_FileNames.tolist()]
    print('For class', currentCls)
    print('Total images:', len(allFileNames))
    print('For Training:', len(train_FileNames))
    print('For Validation:', len(val_FileNames))
    print('For Testing:', len(test_FileNames), '\n')

    def process_images(file_names, destination_dir, current_class):
        accumulator = 1
        for name in file_names:
            new_name = os.path.join(destination_dir, current_class, f"{accumulator}.jpg")
            if os.path.exists(new_name):
                os.remove(new_name)
            img = io.imread(name, as_gray=False)
            roi = resize(img, (100, 100), anti_aliasing=True)
            roi_uint8 = img_as_ubyte(roi)
            io.imsave(new_name, roi_uint8)
            accumulator += 1

    process_images(train_FileNames, os.path.join(final_dir, 'train_val_test_split', 'train'), currentCls)
    process_images(val_FileNames, os.path.join(final_dir, 'train_val_test_split', 'val'), currentCls)
    process_images(test_FileNames, os.path.join(final_dir, 'train_val_test_split', 'test'), currentCls)


src_path = 'C:/D Drive/College/rahul/Natural Calamity Detection - Group Project/Finalized Dataset'
split_dir = os.path.join(src_path, 'train_val_test_split')
if os.path.exists(split_dir):
    shutil.rmtree(split_dir)
os.makedirs(split_dir)

for i in ['Fire_Disaster', 'Land_Disaster', 'Water_Disaster']:
    directory_creation(i, split_dir)
    create_partition(src_path, i, src_path)

print("Running Time --->", "".join([f"{int(x)} {u}" for x, u in zip([divmod(process_time(), 3600)[0], divmod(process_time(), 60)[0], divmod(process_time(), 1)[0], divmod(process_time(), 0.001)[0] - divmod(process_time(), 1)[0]*1000], ["hours, ", "minutes, ", "seconds and ", "milliseconds"]) if x > 0]), "("+str(round(process_time(), 3))+" seconds).")
