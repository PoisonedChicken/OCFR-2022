from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
from scipy import misc
import sys
import os
import imageio
import argparse
#import tensorflow as tf
import pandas as pd
from align_trans import get_reference_facial_points, warp_occlusion
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from util import detect_face, face_preprocess, face_image
import cv2
from PIL import Image, ImageChops 


occlusion_types = ["lower_face","top_face","upper_face","vertical","eye"]

occlusions_combinations = {1:["lower_face"],
                            2:["upper_face"],
                            3:["top_face"],
                            4:["eye"],
                            5:["upper_face","lower_face"],
                            6:["lower_face","top_face"],
                            7:["eye","lower_face"],
                            8:["eye","lower_face","top_face"],
                            9:["upper_face","lower_face","top_face"],
                            10:["top_face","upper_face"],
                            11:["top_face","eye"]}


def get_lmk_from_file(image,lmk_path):
    """
    Gets landmarks for a particular image in rfw dataset, given lmk file and image 
    :param image:  image to extract lmk
    :param lmk_path: path to lmk file
    """

    with open(lmk_path) as f:
        lines = f.readlines()

        for line in lines:
            line_list = line.split('\t')
            line_image = line_list[0].split('/')[-1]

            if line_image == image:

                if '\n' in line_list[-1]:
                    line_list[-1] = line_list[-1][:-1]

                lmks = np.array([[line_list[2],line_list[3]],
                        [line_list[4],line_list[5]],
                        [line_list[6],line_list[7]],
                        [line_list[8],line_list[9]],
                        [line_list[10],line_list[11]]])

                return lmks

def select_occlusion_type():
    return np.random.choice([1,2,4,6,7,10,11]) 

def select_occlusions(occlusions_info):
    types = occlusions_combinations[select_occlusion_type()]
    occlusions = []
    for type in types: 
        occlusions.append(occlusions_info[occlusions_info["type"] == type].sample())
    return occlusions

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def IOU(Reframe,GTframe):
  x1 = Reframe[0]
  y1 = Reframe[1]
  width1 = Reframe[2]-Reframe[0]
  height1 = Reframe[3]-Reframe[1]

  x2 = GTframe[0]
  y2 = GTframe[1]
  width2 = GTframe[2]-GTframe[0]
  height2 = GTframe[3]-GTframe[1]

  endx = max(x1+width1,x2+width2)
  startx = min(x1,x2)
  width = width1+width2-(endx-startx)

  endy = max(y1+height1,y2+height2)
  starty = min(y1,y2)
  height = height1+height2-(endy-starty)

  if width <=0 or height <= 0:
    ratio = 0
  else:
    Area = width*height
    Area1 = width1*height1
    Area2 = width2*height2
    ratio = Area*1./(Area1+Area2-Area)
  return ratio


def main(args):
    random.seed(42)
    np.random.seed(42)
    occlusions_info = pd.read_csv("./occluders/occlusions.csv",names=["path","type","width","height","left_eye","right_eye","nose","left_mouth","right_mouth"])
    #facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    for sub in [""]:#sub in next(os.walk(args.input_dir))[1]:
      print(sub)
      dataset = face_image.get_dataset_common(args.input_dir)
      print(args.input_dir)

      print('dataset size', 'lfw', len(dataset))
      
      print('Creating networks and loading parameters')
      
      with tf.Graph().as_default():
          #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
          #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
          sess = tf.Session()
          with sess.as_default():
              pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
      
      minsize = 20
      threshold = [0.6,0.7,0.9]
      factor = 0.85

      # Add a random key to the filename to allow alignment using multiple processes
      #random_key = np.random.randint(0, high=99999)
      #bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
      #output_filename = os.path.join(output_dir, 'faceinsight_align_%s.lst' % args.name)

      if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

      output_filename = os.path.join(args.output_dir, 'lst_.txt')
      #mask_ftx=open( os.path.join(args.output_dir_mask, 'mask.txt'),'w')
      #masklog=open( os.path.join(args.output_dir_mask, 'lfw-log.txt'),'w')

      
      with open(output_filename, "w+") as text_file:
          nrof_images_total = 0
          nrof = np.zeros( (5,), dtype=np.int32)
          for fimage in dataset:
              _bbox = None
              
              #print(fimage)
              if nrof_images_total%100==0:
                print("Processing %d, (%s)" % (nrof_images_total, nrof))
              nrof_images_total += 1
              #if nrof_images_total<950000:
              #  continue
              image_path_2 =  r'{}'.format(fimage.image_path)
              image_path = image_path_2.replace("\\","/")

              if not os.path.exists(image_path):
                print('image not found (%s)'%image_path)
                continue
              filename = os.path.splitext(os.path.split(image_path)[1])[0]
              #print(image_path)
              try:
                  img = imageio.imread(image_path)
              except (IOError, ValueError, IndexError) as e:
                  errorMessage = '{}: {}'.format(image_path, e)
                  print(errorMessage)
              else:
                  if img.ndim<2:
                      print('Unable to align "%s", img dim error' % image_path)
                      #text_file.write('%s\n' % (output_filename))
                      continue
                  if img.ndim == 2:
                      img = to_rgb(img)
                  img = img[:,:,0:3]
                  _paths = image_path.split('/')
                  
                  a,b = _paths[-2], _paths[-1]

                  target_dir = args.output_dir + '/OCC/' + a
                  target_dir2 = args.output_dir + "_u" + '/' + a
                  target_dir3 = args.output_dir + "_mask"+ '/' + a
                  target_dir4 = args.output_dir + "_mask_u"+ '/' + a
                  target_dir5 = args.output_dir + "_occ_u"+ '/' + a
                  if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                  if not os.path.exists(target_dir2):
                    os.makedirs(target_dir2)
                  if not os.path.exists(target_dir3):
                    os.makedirs(target_dir3)
                  if not os.path.exists(target_dir4):
                    os.makedirs(target_dir4)
                  if not os.path.exists(target_dir5):
                    os.makedirs(target_dir5)
                  target_file = target_dir + '/' + b
                  target_file2 = target_dir2 + '/' + b
                  target_file3 = target_dir3 + '/' + b
                  target_file4 = target_dir4 + '/' + b
                  target_file5 = target_dir5 + '/' + b
                  _minsize = minsize
                  
                  _landmark = None
                  bounding_boxes, points = detect_face.detect_face(img, _minsize, pnet, rnet, onet, threshold, factor)
                  nrof_faces = bounding_boxes.shape[0]


                    


                  
                  if nrof_faces>0:
                    det = bounding_boxes[:,0:4]
                    img_size = np.asarray(img.shape)[0:2]
                    bindex = 0
                    if nrof_faces>1:
                        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                        img_center = img_size / 2
                        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                        bindex = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                    _bbox = bounding_boxes[bindex, 0:4]
                    #_landmark = points[:, bindex].reshape( (2,5) ).T
                    _landmark = get_lmk_from_file(image=b, lmk_path = args.lmk)
                    nrof[0]+=1
                  
                  else:
                    #no Face Detected
                    nrof[1]+=1
                  

                    _landmark = get_lmk_from_file(image=b, lmk_path = args.lmk)
                    # Get image dimensions
                    height, width, _ = np.asarray(img.shape)

                    # Create a bounding box for the entire image
                    x, y, w, h = 0, 0, width, height
                    _bbox = [x, y, w, h]

                    print('No Faces detected in '+a+'/'+b)

                    

                  selected_occlusions = select_occlusions(occlusions_info)
                  warped_occlusion = None
                  

                  img = Image.fromarray(img)
                  im2 = img.copy().convert('L').point( lambda p: 0 ).convert('1') #image to keep just the mask
                  im3 = Image.new("RGB", img.size, color="black") #image to keep just the mask in RGB 

                  



                  for occlusion in selected_occlusions:
                      
                      warped_occlusion = None
                      occlusion_img = Image.open("./occluders/" + str(occlusion.type.iloc[0])+"/"+str(occlusion.path.iloc[0]))
                      if str(occlusion.type.iloc[0]) == "top_face":
                          
                          left_eye = np.float32(occlusion.left_eye.iloc[0].split(","))
                          right_eye = np.float32(occlusion.right_eye.iloc[0].split(","))
                          nose = np.float32(occlusion.nose.iloc[0].split(","))
                          variance = 0#(right_eye[0] - left_eye[0])/5
                          left_eye[0] = np.random.normal(left_eye[0], variance, 1)[0]
                          right_eye[0] = np.random.normal(right_eye[0], variance, 1)[0]

                          warped_occlusion = warp_occlusion(np.array(occlusion_img), [left_eye,right_eye,nose], _landmark[:3],np.array(img).shape)
                  
                      if str(occlusion.type.iloc[0]) == "eye":
                          
                          left_eye = np.float32(occlusion.left_eye.iloc[0].split(","))
                          right_eye = np.float32(occlusion.right_eye.iloc[0].split(","))


                          warped_occlusion = warp_occlusion(np.array(occlusion_img), [left_eye,right_eye], _landmark[:2],np.array(img).shape)

                      if str(occlusion.type.iloc[0]) == "upper_face":
                          left_eye = np.float32(occlusion.left_eye.iloc[0].split(","))
                          right_eye = np.float32(occlusion.right_eye.iloc[0].split(","))
                          nose = np.float32(occlusion.nose.iloc[0].split(","))
                          variance = 0#(right_eye[0] - left_eye[0])/5
                          left_eye[0] = np.random.normal(left_eye[0], variance, 1)[0]
                          right_eye[0] = np.random.normal(right_eye[0], variance, 1)[0]

                          warped_occlusion = warp_occlusion(np.array(occlusion_img), [left_eye,right_eye,nose], _landmark[:3],np.array(img).shape)

                      if str(occlusion.type.iloc[0]) == "lower_face":
                          left_mouth = np.float32(occlusion.left_mouth.iloc[0].split(","))
                          right_mouth = np.float32(occlusion.right_mouth.iloc[0].split(","))
                          nose = np.float32(occlusion.nose.iloc[0].split(","))
                          variance = (right_mouth[0] - left_mouth[0]) * 0.1
                          left_mouth[0] += np.random.normal(0, 0.5, 1) * variance
                          right_mouth[0] += np.random.normal(0, 0.5, 1) * variance

                          #left_mouth[1] = np.random.normal(left_mouth[1], variance, 1)[0]
                          #right_mouth[1] = np.random.normal(right_mouth[1], variance, 1)[0]

                          warped_occlusion = warp_occlusion(np.array(occlusion_img), [nose,left_mouth,right_mouth], _landmark[2:],np.array(img).shape)
                          
                      
                      warped_occlusion = Image.fromarray(warped_occlusion)


                      im2 = ImageChops.add(im2,warped_occlusion.convert('L').point( lambda p: 255 if p > 0 else 0 ).convert('1'))


                      im3 = Image.alpha_composite(im3.convert('RGBA'), warped_occlusion)


                      img = Image.alpha_composite(img.convert('RGBA'), warped_occlusion)


                      
                      #img = Image.fromarray(img)
                      #bgr = img[...,::-1]
                  
                  img = np.asarray(img.convert("RGB"))
                  im2 = np.asarray(im2.convert("RGB"))
                  #
                  warped = face_preprocess.preprocess(img, bbox=_bbox, landmark = _landmark, image_size=args.image_size)
                  warped_2 = face_preprocess.preprocess(im2, bbox=_bbox, landmark = _landmark, image_size=args.image_size)

                  bgr = warped[...,::-1]
                  original_bgr = img[...,::-1]
                  #print(bgr.shape)
                  #print(target_file)



                  cv2.imwrite(target_file, bgr)
                  cv2.imwrite(target_file2 , original_bgr)
                  cv2.imwrite(target_file3 , warped_2[...,::-1])
                  cv2.imwrite(target_file4 , im2[...,::-1])
                  cv2.imwrite(target_file5 , np.asarray(im3.convert("RGB"))[...,::-1])

                  oline = '%s\t%s\t%s\t%s\t%s\n' % (target_file, _bbox[0], _bbox[1], _bbox[2], _bbox[3])
                  text_file.write(oline)
                  

                



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input-dir', type=str, default='Data/lfw' , help='Directory with unaligned images.')
    parser.add_argument('--output-dir', type=str,default='Data/lfw_aligned', help='Directory with aligned face thumbnails.')

    parser.add_argument('--image-size', type=str, help='Image size (height, width) in pixels.', default='112,112')
    parser.add_argument('--lmk', type=str,default='', help='path to LMKs to use when no face detected')
    #parser.add_argument('--margin', type=int,
    #    help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


