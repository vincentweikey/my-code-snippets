import os
 
def get_all_file_name(file_dir, extnames):
    """Iterate through all files in the folder with an extension of extnames List
    """
    all_file_names = []
    for dirs, dir_names, files in os.walk(file_dir):
        for f in files:
            ext = os.path.splitext(f)[-1]
            if ext.lower() in extnames:
                file_name = os.path.abspath(os.path.join(dirs, f))
                all_file_names.append(file_name)
    return all_file_names

def get_all_image_name(file_dir):
    return get_all_file_name(file_dir, ['.jpg','.png','.jpeg','.bmp'])

def get_all_video_name(file_dir):
    return get_all_file_name(file_dir, ['.mov','.mkv','.mp4','.avi'])

def create_foldr_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    else:
        pass
        
import hashlib

def get_file_md5(file_name):
    """[calculate file hash md5]
    """
    if os.path.isfile(file_name):
        file_content = open(file_name, 'rb')
        contents = file_content.read()
        file_content.close()
        md5 = hashlib.md5(contents).hexdigest()
    else:
        md5 = None
    return md5

import cv2
def rename_image_as_md5_and_united_in_new_folder(root, dst_folder, save_ext='.jpg'):
    """[just as name]
    Args:
        root ([type]): [router root]
        dst_folder ([type]): [where to save]
        save_ext (str, optional): [ext of save format]. Defaults to '.jpg'.
    """
    for image_path in get_all_image_name(root):
        md5 =  get_file_md5(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        save_name = os.path.join(dst_folder, md5+save_ext)
        cv2.imwrite(save_name, img, [int( cv2.IMWRITE_JPEG_QUALITY), 100])

