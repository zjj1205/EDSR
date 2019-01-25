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
def ed(file_dir,out_dir):

    args = parser.parse_args()
    for item in os.listdir(file_dir):
        file_dir1 = os.path.join(file_dir, item)
        out_dir1 = os.path.join(out_dir, item)
        if not os.path.exists(out_dir1):
            os.mkdir(out_dir1)
        if os.path.isdir(file_dir1):
            for item1 in os.listdir(file_dir1):
                file_dir2 = os.path.join(file_dir1, item1)
                out_dir2 = os.path.join(out_dir1, item1)
                if not os.path.exists(out_dir2):
                    os.mkdir(out_dir2)
                if os.path.isdir(file_dir2):
                    for item2 in os.listdir(file_dir2):
                        tf.reset_default_graph()
                        final_dir = os.path.join(out_dir2,item2)
                        image = os.path.join(file_dir2, item2)
                        down_size = args.imgsize // args.scale
                        network = EDSR(down_size, args.layers, args.featuresize, scale=args.scale)

                        network.resume(args.savedir)
                        x = scipy.misc.imread(image)
                        # t0 = time.time()
                        outputs = network.predict(x)
                        # print(time.time() - t0, 5555555555)
                        if len(image) > 0:
                            scipy.misc.imsave(final_dir,  outputs)

file_dir = '/tmp/pycharm_project_293/knn/data/shentongtest/'
out_dir = '/tmp/pycharm_project_496/out'
ed(file_dir, out_dir)