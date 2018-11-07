# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:08:18 2018

@author: jimena
"""
import sys, os
from PIL import Image
import uuid

def ParseData(landmark_name, out_dir_resize, out_dir_borders, image_dir):
  countTot = 0
  desired_size = 512
  for item in os.listdir(image_dir):
          countTot = countTot +1
          # Creates a new name for the image that will be use in all types of image storage, in order to keep a track between the original image and their different formats
          nameFile =  landmark_name + str(uuid.uuid4()) + '.jpg'          
          # Renames the original image with the new name
          os.rename(image_dir + item, image_dir + nameFile)
          # Resize the image to 512 x 512 using Lanczos filter
          image = Image.open(image_dir + nameFile)
          image = image.resize((desired_size, desired_size), Image.LANCZOS)          
          image.save(os.path.join(out_dir_resize, nameFile))    
          # First resize the input image so that its maximum size equals to the given size. Then we pad the resized image to make it square
          imageB = Image.open(image_dir + nameFile)
          old_size = imageB.size
          ratio = float(desired_size)/max(old_size)
          new_size = tuple([int(x*ratio) for x in old_size])
          imageB = imageB.resize(new_size, Image.LANCZOS)
          new_im = Image.new("RGB", (desired_size, desired_size))
          new_im.paste(imageB, ((desired_size-new_size[0])//2,(desired_size-new_size[1])//2))
          new_im.save(os.path.join(out_dir_borders, nameFile))
          
  print(' TOTAL IMAGENES: ', countTot)
    
    
def Run():
  if len(sys.argv) != 5 :
    print('Syntax: %s <landmark_name> <out_dir_resize/> <out_dir_borders/> <image_dir/>' % sys.argv[0])
    sys.exit(0)
  (landmark_name, out_dir_resize, out_dir_borders, image_dir) = sys.argv[1:]

  ParseData(landmark_name, out_dir_resize, out_dir_borders, image_dir)
  
if __name__ == '__main__':
  Run()