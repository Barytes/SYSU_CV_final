import os
import shutil

src_path = "../raw_data/"
dest_path = "../data/train/"
list = os.listdir("../raw_data/gt")
for f in list:
    if int(f.split('.')[0]) % 100 != 57:
        # shutil.copyfile(src_path+"gt/"+f, dest_path+"gt/"+f)
        # shutil.copyfile(src_path+"imgs/"+f, dest_path+"imgs/"+f)
        shutil.copyfile(src_path+"gt/"+f, dest_path+"gt/"+f)
        shutil.copyfile(src_path+"imgs/"+f, dest_path+"imgs/"+f)