import data
import argparse
from model import EDSR
import numpy as np
import os
import scipy.misc
import random
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="/tmp/pycharm_project_496/EDSR-Tensorflow-master/data/General-100")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=10,type=int)
parser.add_argument("--savedir",default='/tmp/pycharm_project_496/EDSR-Tensorflow-master/saved_models')
parser.add_argument("--iterations",default=1000,type=int)
args = parser.parse_args()
data.load_dataset(args.dataset,args.imgsize)
# data_dir = args.dataset
# img_size = args.imgsize
# # def load_dataset(data_dir, img_size):
# global train_set
# global test_set
# imgs = []
# img_files = os.listdir(data_dir)
# for img in img_files:
# 	try:
# 		tmp= scipy.misc.imread(data_dir+"/"+img)
# 		x,y,z = tmp.shape
# 		coords_x = x / img_size
# 		coords_y = y / img_size
# 		coords = [ (q,r) for q in range(int(coords_x)) for r in range(int(coords_y)) ]
# 		for coord in coords:
# 			imgs.append((data_dir+"/"+img,coord))
# 	except:
# 		print("oops")
# test_size = min(10,int(len(imgs)*0.2))
# random.shuffle(imgs)
# test_set = imgs[:test_size]
# train_set = imgs[test_size:][:200]
# print(test_set,train_set,1111111)
# return



if args.imgsize % args.scale != 0:
    print(f"Image size {args.imgsize} is not evenly divisible by scale {args.scale}")
    # return
down_size = args.imgsize//args.scale
# print(data.load_dataset(args.dataset,args.imgsize),11121)
network = EDSR(down_size,args.layers,args.featuresize,args.scale)
network.set_data_fn(data.get_batch,(args.batchsize,args.imgsize,down_size),data.get_test_set,(args.imgsize,down_size))
network.train(args.iterations,args.savedir)
