import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable

from net import Net
#from option import Options
#import utils
from utils import StyleLoader

#args in args.subcommand = “demo"

style_folder="images/9styles/"
style_size=128
cuda=1
#record=0
model21="models/21styles.model"
this_ngf=128 #for the NET
demo_size=210

#def run_demo(args, mirror=False):
def run_demo():
	style_model = Net(ngf=this_ngf)
	model_dict = torch.load(model21)
	model_dict_clone = model_dict.copy()

	for key, value in model_dict_clone.items(): #v0.4
		if key.endswith(('running_mean', 'running_var')): #v0.4
			del model_dict[key]	

	style_model.load_state_dict(model_dict, True)

	style_model.eval()
	#if cuda:
	style_loader = StyleLoader(style_folder, style_size)
	style_model.cuda()
	#else:
	#	style_loader = StyleLoader(style_folder, style_size, False)

	# Define the codec and create VideoWriter object
	height =  demo_size
	width = int(4.0/3*demo_size)
	swidth = int(width/4)
	sheight = int(height/4)
	#if record:
	#	fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
	#	out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (2*width, height))
	cam = cv2.VideoCapture(0)
	cam.set(3, width)
	cam.set(4, height)
	key = 0
	idx = 0
	while True:
		# read frame
		idx += 1
		ret_val, img = cam.read()
		#if mirror: 
		#	img = cv2.flip(img, 1)
		#cimg = img.copy()###############################OR-CAM
		img = np.array(img).transpose(2, 0, 1)
		# changing style 
		if idx%20 == 1:
			style_v = style_loader.get(int(idx/20))
			style_v = Variable(style_v.data)
			style_model.setTarget(style_v)

		img=torch.from_numpy(img).unsqueeze(0).float()
		#if cuda:
		img=img.cuda()

		img = Variable(img)
		img = style_model(img)

		#if cuda:
		#simg = style_v.cpu().data[0].numpy()
		img = img.cpu().clamp(0, 255).data[0].numpy()
		#else:
		#	simg = style_v.data.numpy()
		#	img = img.clamp(0, 255).data[0].numpy()
		img = img.transpose(1, 2, 0).astype('uint8')
		#simg = simg.transpose(1, 2, 0).astype('uint8')

		# display
		#simg = cv2.resize(simg,(swidth, sheight), interpolation = cv2.INTER_CUBIC)
		#cimg[0:sheight,0:swidth,:]=simg
		#img = np.concatenate((cimg,img),axis=1)
		cv2.imshow('MSG Demo', img)

		#cv2.imwrite('stylized/%i.jpg'%idx,img)
		key = cv2.waitKey(1)
		#if args.record:
		#	out.write(img)
		if key == 27: 
			break

	cam.release()
	#if args.record:
	#	out.release()
	cv2.destroyAllWindows()

def main():
	# getting things ready
	#args = Options().parse()
	#if args.subcommand is None:
	#	raise ValueError("ERROR: specify the experiment type")
	#if args.cuda and not torch.cuda.is_available():
	#	raise ValueError("ERROR: cuda is not available, try running on CPU")

	# run demo
	#run_demo(args, mirror=True)
	run_demo()

if __name__ == '__main__':
	main()
