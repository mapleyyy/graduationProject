#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 16:26:04 2017

@author: gaoyufeng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:19:32 2017

@author: gaoyufeng
"""
TRAIN_NORMAL_WSI_PATH = '/Users/gaoyufeng/毕设/data/normal/'
TRAIN_TUMOR_WSI_PATH = '/Users/gaoyufeng/毕设/data/tumor/'
TRAIN_TUMOR_MASK_PATH = '/Users/gaoyufeng/毕设/data/mask/'

PATCHES_NORMAL_NEGATIVE_PATH = '/Users/gaoyufeng/毕设/data/normal_negative_patch/'
PATCHES_TUMOR_NEGATIVE_PATH = '/Users/gaoyufeng/毕设/data/tumor_negative_patch/'
PATCHES_POSITIVE_PATH = '/Users/gaoyufeng/毕设/data/tumor_positive_patch/'
PATCHES_FROM_USE_MASK_POSITIVE_PATH = '/Users/gaoyufeng/毕设/data/mask_positive_patch/'

from PIL import Image
import openslide
import numpy as np
import cv2
import glob

PATCH_SIZE = 256

'''
    命名规则：
    图像分为normal，tumor和mask；
    normal和tumor相关变量以wsi_开头；
    mask相关变量以mask_开头；
    后面跟：
    1.特殊申明；
    2.数据特殊类型；
    rgb 代表 RGB数据
    rgba 代表 RGBA数据
    bin 代表 二值图像数据
    hsv 代表 HSV类型数据
    arr 代表 np.array数组
    pil 代表 PIL格式图像数据
'''
class WSI(object):
    
    negative_index = 1
    positive_index = 1
    '''
    类构造函数
    WSI类，用来处理WSI图片
    输入：WSI图片
    输出：标记为positive和negative的patch
    '''
    def __init__(self,patch_size = 256):
        self.patch_size = patch_size # 设置patch的大小
        self.init_level = 7 # 由于size太大无法读入较高level的图片，所以初始的level比较低
        
    
    '''
    读取normal图像函数
    @param：
        wsi_path：wsi图片路径
    '''
    def read_wsi_normal(self,wsi_path):
        try:
            self.wsi_path = wsi_path
            self.wsi_handle = openslide.OpenSlide(self.wsi_path) # 获得tif图像的句柄
            
            self.used_level = min(self.init_level , self.wsi_handle.level_count - 1)
            self.wsi_rgba_pil = self.wsi_handle.read_region((0,0),self.used_level,
                                                           self.wsi_handle.level_dimensions[self.used_level]) # 获取used_level层级的图像
            
            self.wsi_rgba_pil.show()
            self.wsi_rgba_arr = np.array(self.wsi_rgba_pil) # 将PIL类型的数据转换成array型的数据格式
            
            
            self.wsi_rgb_arr = cv2.cvtColor(self.wsi_rgba_arr, cv2.COLOR_RGBA2RGB) # a通道对我们没有作用，就转成rgb格式的
            
            self.wsi_handle.close()
        except openslide.OpenSlideError:
            print('Error:OpenSlideError')
            return False
        
        return True
    
    
    '''
    读取tumor和mask图像函数，因为一般都是要一起读入的
    @param：
        wsi_path：wsi图片的路径
        mask_path：mask图片的路径
    '''
    def read_wsi_tumor_mask(self,wsi_path,mask_path):
        try:
            self.wsi_path = wsi_path
            self.mask_path = mask_path
            self.wsi_handle = openslide.OpenSlide(self.wsi_path)
            self.mask_handle = openslide.OpenSlide(self.mask_path) # 分别获取两个tif图像句柄
            
            self.used_level = min(self.init_level , self.wsi_handle.level_count - 1 , self.mask_handle.level_count - 1)
            # print(self.used_level)
            
            self.wsi_rgba_pil = self.wsi_handle.read_region((0,0),self.used_level,
                                                           self.wsi_handle.level_dimensions[self.used_level]) # 获取used_level层级的图像
            self.mask_rgba_pil = self.mask_handle.read_region((0,0),self.used_level,
                                                           self.wsi_handle.level_dimensions[self.used_level])
            
            # self.wsi_rgba_pil.show()
            # self.mask_rgba_pil.show()
            
            self.wsi_rgba_arr = np.array(self.wsi_rgba_pil)
            self.mask_rgba_arr = np.array(self.mask_rgba_pil)
            
            self.wsi_rgb_arr = cv2.cvtColor(self.wsi_rgba_arr, cv2.COLOR_RGBA2RGB)
            self.mask_rgb_arr = cv2.cvtColor(self.mask_rgba_arr, cv2.COLOR_RGBA2RGB)
            # print(self.wsi_rgb_arr.shape)
            # print(self.mask_rgb_arr.shape)
            self.wsi_handle.close()
            self.mask_handle.close()
            
        except openslide.OpenSlideError:
            print('Error:OpenSlideError')
            return False
        return True
    
    
    '''
    处理normal和tumor图片的前后景分离
    @para：
        wsi_type：图片类型 normal或者tumor
    '''
    def find_wsi_roi(self,wsi_type):
        if wsi_type == 'normal':
            self.lower_red = np.array([20, 50, 20])
            self.upper_red = np.array([179, 150, 200])
        else:
            self.lower_red = np.array([20, 20, 20])
            self.upper_red = np.array([179, 255, 255])

        wsi_hsv_arr = cv2.cvtColor(self.wsi_rgb_arr, cv2.COLOR_RGB2HSV) # 将rgb图像转换为hsv
        wsi_bin_arr = cv2.inRange(wsi_hsv_arr,self.lower_red,self.upper_red) # wsi_bin_arr 通道为1
        
        close_kernel = np.ones((25, 25), dtype=np.uint8)
        wsi_close_pil = Image.fromarray(cv2.morphologyEx(np.array(wsi_bin_arr), cv2.MORPH_CLOSE, close_kernel))
        
        open_kernel = np.ones((30, 30), dtype=np.uint8)
        wsi_open_pil = Image.fromarray(cv2.morphologyEx(np.array(wsi_close_pil), cv2.MORPH_OPEN, open_kernel)) # 对于获得的二值图像进行形态学的开闭操作，使二值部分相对连续
        self.wsi_mask_pil = wsi_open_pil
        self.wsi_mask_pil.show('NOT OTSU')
        
    
    '''
    获得wsi图片的bounding数据
    @ret：
        bounding_box：边界信息数组，x y w h
    '''
    def get_wsi_roi_bounding(self):
        image,contours, hierarchy = cv2.findContours(np.array(self.wsi_mask_pil),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
        wsi_rgb_arr_tmp = self.wsi_rgb_arr
        cv2.drawContours(wsi_rgb_arr_tmp,contours,-1,(0,255,0),2) # cv2函数作用对象都是数组，所以不管是rgb还是bgr，都是按通道计算
        # self.display_wsi(wsi_rgb_arr_tmp)
        
        bounding_box = []
        for cnt in contours:
            bounding_box.append(cv2.boundingRect(cnt)) #bounding rect 四个数据分别为x y w h
        return bounding_box
        
    '''
    获得mask图片的bounding数据
    @ret：
        bounding_box：边界信息，x y w h
    '''
    def get_mask_roi_bounding(self):
        mask_bin_arr = cv2.cvtColor(self.mask_rgb_arr,cv2.COLOR_RGB2GRAY)
        # print(mask_bin_arr.shape)
        image,contours, hierarchy = cv2.findContours(mask_bin_arr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_box = []
        for cnt in contours:
            bounding_box.append(cv2.boundingRect(cnt)) #bounding rect 四个数据分别为x y w h
        return bounding_box
    
        
        
    '''
    用otsu算法分离图片前后景
    '''
    def find_wsi_roi_otsu(self):
        wsi_gray_arr = cv2.cvtColor(self.wsi_rgb_arr, cv2.COLOR_RGB2GRAY)
        threshold,wsi_bin_otsu = cv2.threshold(wsi_gray_arr,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        print(threshold)
        # self.display_wsi(self.wsi_rgb_arr)
        # self.display_wsi(wsi_gray_arr)
        self.display_wsi(wsi_bin_otsu, 'BIN')
        close_kernel = np.ones((25, 25), dtype=np.uint8)
        wsi_close_arr = cv2.morphologyEx(np.array(wsi_bin_otsu), cv2.MORPH_CLOSE, close_kernel)
        # self.display_wsi(wsi_close_arr, 'BIN')

        open_kernel = np.ones((30, 30), dtype=np.uint8)
        wsi_open_bin = cv2.morphologyEx(np.array(wsi_close_arr), cv2.MORPH_OPEN, open_kernel)
        self.display_wsi(wsi_open_bin, 'BIN')
        
    
    ''' 
    测试用显示图片函数
    @param：
        img_arr：图像数组 np.array
        img_type：图像类型
    '''
    def display_wsi(self, img_arr,img_type = 'RGB'):
        if img_type == 'HSV':
            tmp_arr = cv2.cvtColor(img_arr, cv2.COLOR_HSV2RGB)
            tmp_img = Image.fromarray(tmp_arr)
            tmp_img.show()
        elif img_type == 'GRAY':
            tmp_img = Image.fromarray(img_arr)
            tmp_img.show()
        elif img_type == 'BIN':
            tmp_img = Image.fromarray(img_arr)
            tmp_img.show()
        else:
            tmp_img = Image.fromarray(img_arr)
            print(tmp_img.mode)
            tmp_img.show()
    
    def add_index(self):
        print(WSI.negative_index)
        WSI.negative_index += 1
            
 
def run_on_tumor_data():
    wsi_path_arr = glob.glob(TRAIN_TUMOR_WSI_PATH + '*.tif')
    mask_path_arr = glob.glob(TRAIN_TUMOR_MASK_PATH + '*.tif')
    wsi_path_arr.sort()
    mask_path_arr.sort()
    print(wsi_path_arr)
    
    wsi = WSI()
    
    for wsi_path, mask_path in zip(wsi_path_arr,mask_path_arr):
        if wsi.read_wsi_tumor_mask(wsi_path, mask_path):
            # wsi.find_wsi_roi_otsu()
            # wsi.find_wsi_roi('tumor')
            # bounding = wsi.get_wsi_roi_bounding()
            pass



def run_on_mask_data():
    wsi_path_arr = glob.glob(TRAIN_TUMOR_WSI_PATH + '*.tif')
    mask_path_arr = glob.glob(TRAIN_TUMOR_MASK_PATH + '*.tif')
    wsi_path_arr.sort()
    mask_path_arr.sort()
    
    wsi = WSI()
    
    for wsi_path, mask_path in zip(wsi_path_arr,mask_path_arr):
        if wsi.read_wsi_tumor_mask(wsi_path, mask_path):
            # bounding = wsi.get_mask_roi_bounding()
            pass
    

def run_on_normal_data():
    wsi_path_arr = glob.glob(TRAIN_NORMAL_WSI_PATH + '*.tif')
    wsi_path_arr.sort()
       
    wsi = WSI()
    
    for wsi_path in wsi_path_arr:
        if wsi.read_wsi_normal(wsi_path): # 读取WSI数据
            # wsi.find_wsi_roi('normal') # 分离图像前景和背景，获得WSI图像的mask图像
            # bounding = wsi.get_wsi_roi_bounding() # 获取ROI的边框
            pass
            
        
        
        
        
if __name__=='__main__':
    # run_on_normal_data()
    run_on_tumor_data()
    # run_on_mask_data()
    
    
    
    
    
    
    