from typing import Counter
import numpy as np
from PIL import Image
import os
import glob
from PIL import Image, ImageOps
from multiprocessing import Process, Value, Lock, Pool
import time

class ImageProcess:
    def __init__(self, json_path, save_path, rgb_path, gray_path, black_threshold = 25):
        self.black_threshold = black_threshold
        self.gray_files = None
        self.rgb_files = None
        self.save_path = None
        self.origin_json = None
        self.json  = {
         "count": 0 ,
         "lon":[],
         "lat":[],
         "idx":[]
        }
        self.sum_threshold = 10000
        self.counter = 0
        self.load_paths(rgb_path=rgb_path, gray_path=gray_path)
        self.set_save_path(save_path)
        self.set_json(json_path)
    
    def load_paths(self, rgb_path, gray_path):
        self.rgb_files = glob.glob(rgb_path)
        self.gray_files = glob.glob(gray_path)

    def set_save_path(self, path):
        self.save_path = path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    def set_json(self, path):
        import json
        with open(path, "r") as fp:
            self.origin_json = json.load(fp)
        self.json  = {
         "count": 0 ,
         "lon":[],
         "lat":[],
         "idx":[],
         "before_idx":[]
        } 
    def set_black_thresold(self, threshold):
        self.black_threshold = threshold

    def merge_without_sea(self, idx):
        sat_img = Image.open(self.rgb_files[idx])
        sat_img = sat_img.convert("RGBA")
        sat_arr = np.array(sat_img)

        gray_img = Image.open(self.gray_files[idx])
        gray_img = ImageOps.grayscale(gray_img)
        gray_arr = np.array(gray_img)

        sum_result = np.sum(gray_arr)
        if sum_result <= self.sum_threshold or np.max(gray_arr) <= self.black_threshold:
            print("skip: ", idx)
            return

        for row in range(sat_arr.shape[0]):
            for col in range(sat_arr.shape[1]):
                sat_arr[row][col][3] = gray_arr[row][col]

        im = Image.fromarray(sat_arr)      
        im.save(f'{self.save_path}/{idx:05d}.png')
    
    def write_json(self, path):
        import json
        with open(path, 'w') as outfile:
            json.dump(self.json, outfile, indent=4)
            
    def start_merge_without_sea(self):
        pool = Pool()
        print(len(self.rgb_files))
        pool.map(self.merge_without_sea, [i for i in range(len(self.rgb_files))])
        pool.close()
        pool.join()

