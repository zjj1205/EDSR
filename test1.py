from model import EDSR
import scipy.misc
import tensorflow as tf
import argparse
import os
import time

parser = argparse.ArgumentParser()
# parser.add_argument("--dataset",default="data/General-100")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=10,type=int)
parser.add_argument("--savedir",default="saved_models")
parser.add_argument("--iterations",default=1000,type=int)
parser.add_argument("--numimgs",default=5,type=int)
# parser.add_argument("--outdir",default="out")
# parser.add_argument("--file_dir",default="c2f57b11-8f82-40d5-b9a4-24a9129e2893.jpg")

# args = parser.parse_args()
# if not os.path.exists(args.outdir):
# 	os.mkdir(args.outdir)
# down_size = args.imgsize//args.scale
# network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale)
# network.resume(args.savedir)
def ed(file_dir,outdir):

    args = parser.parse_args()
    for item in os.listdir(file_dir):
        file_dir1 = os.path.join(file_dir, item)
        outdir1 = os.path.join(outdir, item)
        if not os.path.exists(outdir1):
            os.mkdir(outdir1)
        if os.path.isdir(file_dir1):
            for item1 in os.listdir(file_dir1):
                tf.reset_default_graph()
                final_dir = os.path.join(outdir1,item1)
                image = os.path.join(file_dir1, item1)
                down_size = args.imgsize // args.scale
                network = EDSR(down_size, args.layers, args.featuresize, scale=args.scale)

                network.resume(args.savedir)
                x = scipy.misc.imread(image)
                t0 = time.time()
                outputs = network.predict(x)
                print(time.time() - t0, 5555555555)
                if len(image) > 0:
                    scipy.misc.imsave(final_dir,  outputs)

file_dir = '/tmp/pycharm_project_496/zjj_data/wjq/'
outdir = '/tmp/pycharm_project_496/EDSR-Tensorflow-master/out'
ed(file_dir,outdir)