
import os

# set directory path
dir_path = "C:/Users/ohad/Videos/RWF-2000/train/Fight"


# loop through all files in the directory
for filename in os.listdir(dir_path):
    # check if file ends with .jpg
    if filename.endswith('.jpg'):
        # join directory path and file name to create full file path
        file_path = os.path.join(dir_path, filename)
        # delete the file
        os.remove(file_path)