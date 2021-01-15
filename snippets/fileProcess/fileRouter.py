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