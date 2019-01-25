from model import EDSR
import scipy.misc
import os

dataset = "data/General-100"
imgsize = 100
scale = 2
layers = 32
featuresize = 256
batchsize = 1
savedir = "saved_models"
iterations=1000
numimgs=5
outdir="dataout"
image="1.jpg"

if not os.path.exists(outdir):
	os.mkdir(outdir)
down_size = imgsize//scale
network = EDSR(down_size,layers,featuresize,scale=scale)
network.resume(savedir)
if len(image) > 0:
	x = scipy.misc.imread(image)
else:
	print("No image argument given")
inputs = x
outputs = network.predict(x)
if image:
	scipy.misc.imsave(outdir+"/input_"+image,inputs)
	scipy.misc.imsave(outdir+"/output_"+image,outputs)