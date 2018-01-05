#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 16:19:33 2017

@author: gaoyufeng
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math 
import glob

import tensorflow as tf

import dataset_utils

PATCHES_NORMAL_NEGATIVE_PATH = '/Users/gaoyufeng/毕设/data/normal_negative_patch/'
PATCHES_TUMOR_NEGATIVE_PATH = '/Users/gaoyufeng/毕设/data/tumor_negative_patch/'
PATCHES_POSITIVE_PATH = '/Users/gaoyufeng/毕设/data/tumor_positive_patch/'
PATCHES_FROM_USE_MASK_POSITIVE_PATH = '/Users/gaoyufeng/毕设/data/mask_positive_patch/'

TFRECORD_PATH = '/Users/gaoyufeng/毕设/data/tfrecord/'


# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 2

_NUM_PER_RECORD = 262144

_CLASS_NAMES_TO_IDS = {'negative':0,'positive':1}

'''
用于解析图像的类
'''
class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""
  def __init__(self):
      # Initializes function that decodes RGB JPEG data.
      self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
      self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
    
  def read_image_dims(self, sess, image_data):
      image = self.decode_jpeg(sess, image_data)
      return image.shape[0], image.shape[1]
    
  def decode_jpeg(self, sess, image_data):
      image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
      
      assert len(image.shape) == 3
      assert image.shape[2] == 3
      return image
    
'''
获得TFrecord文件的名称
@param:
    dataset_dir record文件保存的目录
    split_name train/validation 
    shard_id 数据的类别
    record_id record的序号
@ret:
    TFreocrd文件的路径名称
'''
def _get_dataset_filename(dataset_dir, split_name, shard_id, record_id):
    output_filename = 'data_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, record_id)
    return os.path.join(dataset_dir, output_filename)

'''
转换数据，从单个的jpg数据转换到TFrecord
@param:
    split_name train/validation
    filenames jpg数据的路径列表
    dataset_dir record文件保存目录
    class_id 数据的类别
'''
def _convert_dataset(split_name, filenames, dataset_dir,class_id):
  
    record_file_num = math.ceil(len(filenames)/_NUM_PER_RECORD)
    f = open('./log.txt','a')
    with tf.Graph().as_default():
        image_reader = ImageReader()
    
        with tf.Session('') as sess:
        
          for record_id in range(record_file_num):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, class_id, record_id)
        
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        
                for i in range(_NUM_PER_RECORD):
                    if record_id*_NUM_PER_RECORD + i >= len(filenames):
                        break
                    # Read the filename:
                    image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                    height, width = image_reader.read_image_dims(sess, image_data)
        
                    example = dataset_utils.image_to_tfexample(
                        image_data, b'jpg', height, width, class_id)
                    tfrecord_writer.write(example.SerializeToString())
                    print('write '+filenames[i]+' to '+output_filename)
                    f.write('write '+filenames[i]+' to '+output_filename+'\n')
    
    f.close()

  

def run(data_dir, dataset_dir, class_id):
    patch_paths = glob.glob(data_dir+'*.jpg')
    patch_paths.sort()
    # First, convert the training and validation sets.
    _convert_dataset('train', patch_paths, dataset_dir,class_id)
    
def run_on_negative():
    patch_paths = glob.glob(PATCHES_NORMAL_NEGATIVE_PATH+'*.jpg')
    patch_paths2 = glob.glob(PATCHES_TUMOR_NEGATIVE_PATH + '*.jpg')
    patch_paths = patch_paths + patch_paths2
    patch_paths.sort()
    # print(patch_paths)
    
    _convert_dataset('train', patch_paths, TFRECORD_PATH,_CLASS_NAMES_TO_IDS['negative'])

def run_on_positive():
    patch_paths = glob.glob(PATCHES_POSITIVE_PATH+'*.jpg')
    patch_paths2 = glob.glob(PATCHES_FROM_USE_MASK_POSITIVE_PATH + '*.jpg')
    patch_paths = patch_paths + patch_paths2
    patch_paths.sort()
    # print(patch_paths)
    
    _convert_dataset('train', patch_paths, TFRECORD_PATH,_CLASS_NAMES_TO_IDS['positive'])

if __name__ == '__main__':
    run_on_positive()
    run_on_negative()
    
    
    
    
    