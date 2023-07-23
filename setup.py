import os, sys
import shutil
images_dir = './data/images/Images'
listdir = os.listdir(images_dir)
di = []
for folder in listdir:
  #print(folder, len(os.listdir(images_dir + '/' + folder)))
  di.append((folder, len(os.listdir(images_dir + '/' + folder)) ))

di.sort(key=lambda x: x[1], reverse=True)
#!rm -rf /content/data/images/CloneImages
#print(*di[:5], sep="\n")
clone_dir = './data/images/CloneImages'
os.makedirs('./data/images/CloneImages',exist_ok = True)
for folder, _ in di[0:20]:
  shutil.copytree(images_dir + '/' + folder, clone_dir + '/' + folder)
