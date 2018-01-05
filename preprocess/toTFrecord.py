#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:25:35 2017

@author: gaoyufeng
"""

PATCHES_NORMAL_NEGATIVE_PATH = '/Users/gaoyufeng/毕设/data/normal_negative_patch/'
PATCHES_TUMOR_NEGATIVE_PATH = '/Users/gaoyufeng/毕设/data/tumor_negative_patch/'
PATCHES_POSITIVE_PATH = '/Users/gaoyufeng/毕设/data/tumor_positive_patch/'
PATCHES_FROM_USE_MASK_POSITIVE_PATH = '/Users/gaoyufeng/毕设/data/mask_positive_patch/'

TFRECORD_PATH = '/Users/gaoyufeng/毕设/data/tfrecord/'

from PIL import Image
import numpy as np
import cv2
import glob
import tensorflow as tf
import math 

TFRECORD_ITEM_NUM = 10
BATCH_SIZE = 1
NUM_THREADS = 1

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_TFrecord(patch_path, record_path, label):
    patch_path = glob.glob(patch_path+'*.png')
    patch_path.sort()
    print(len(patch_path))
    record_file_num = math.ceil(len(patch_path)/TFRECORD_ITEM_NUM)
    # record_file_num = 1
    for i in range(record_file_num):
        _record_path = record_path + '_' + str(i) + '.tfrecords'
        writer = tf.python_io.TFRecordWriter(_record_path)
        
        for j in range(TFRECORD_ITEM_NUM):
    
            if i*TFRECORD_ITEM_NUM + j >= len(patch_path):
                break
            
            patch_pil = Image.open(patch_path[i*TFRECORD_ITEM_NUM + j])
            # if (i == 0) and (j == 0):
            #     patch_pil.show()
            patch_rgb_arr = cv2.cvtColor(np.array(patch_pil),cv2.COLOR_RGBA2RGB)
            # print(patch_rgb_arr.dtype)
            patch_rgb_raw = patch_rgb_arr.tobytes()
            # print(type(patch_rgb_raw))
            example = tf.train.Example(features = tf.train.Features(feature = {
                "imgdata": _bytes_feature(patch_rgb_raw),
                "label": _int64_feature(label)
                }))
            writer.write(example.SerializeToString())
            print('write ' + str(i*TFRECORD_ITEM_NUM + j) + ' record to ' + _record_path)
            
            
        writer.close()

       
def read_TFrecord(record_path):
    
    file_queue = tf.train.string_input_producer(record_path)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    
    features = tf.parse_single_example(serialized_example,
                                        features = {
                                        'imgdata':tf.FixedLenFeature([], tf.string),
                                        'label':tf.FixedLenFeature([], tf.int64)
                                        })
    
    imgdata = tf.decode_raw(features['imgdata'],tf.uint8)
    imgdata.set_shape(256*256*3)
    imgdata = tf.reshape(imgdata, [256,256,3])
    
    image_batch, label_batch = tf.train.shuffle_batch([imgdata, features['label']],     #<---------多线程随机batch生成
                                            batch_size=BATCH_SIZE,
                                            num_threads=NUM_THREADS,
                                            capacity=100 + 3 * BATCH_SIZE,
                                            min_after_dequeue=100)
    return image_batch, label_batch

            
            

if __name__ == '__main__':
    create_TFrecord(PATCHES_NORMAL_NEGATIVE_PATH,TFRECORD_PATH + 'negative',0)
    
    image_batch,label_batch = read_TFrecord(['/Users/gaoyufeng/毕设/data/tfrecord/negative_0.tfrecords'])
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        image, label = sess.run([image_batch,label_batch])
        print(type(image))
        print(image.shape)
        Image.fromarray(image[0]).show()
        coord.request_stop()           
        coord.join(threads) 
    
    
    
    
    