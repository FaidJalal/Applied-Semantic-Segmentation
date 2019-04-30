import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import imutils

from utils import utils, helpers
from builders import model_builder
checkpoint_path = 'checkpoints/latest_model_BiSeNet_CamVid.ckpt'
crop_height = 512
crop_width = 512
model = 'BiSeNet'
dataset = 'CamVid'
class_names_list, label_values = helpers.get_label_info(os.path.join(dataset, "class_dict.csv"))
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
classes = 'class_colors/class.txt'
colors = 'class_colors/colors.txt'

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", dataset)
print("Model -->", model)
print("Crop Height -->", crop_height)
print("Crop Width -->", crop_width)
print("Num Classes -->", num_classes)
# print("Image -->", img)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, _ = model_builder.build_model(model, net_input=net_input,
	                                num_classes=num_classes,
	                                crop_width=crop_width,
	                                crop_height=crop_height,
	                                is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, checkpoint_path)


def pred(img):
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--image', type=str, default=None, required=True, help='The image you want to predict on. ')
	parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for 		your model.')
	parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
	parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
	parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
	parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')

	'''



	print("Testing image " + img)

	loaded_image = utils.load_image(img)
	resized_image =cv2.resize(loaded_image, (crop_width, crop_height))
	input_image = np.expand_dims(np.float32(resized_image[:crop_height, :crop_width]),axis=0)/255.0

	st = time.time()
	output_image = sess.run(network,feed_dict={net_input:input_image})

	run_time = time.time()-st

	output_image = np.array(output_image[0,:,:,:])
	output_image = helpers.reverse_one_hot(output_image)

	out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
	file_name = utils.filepath_to_name(img)
	res=cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)
	cv2.imwrite("%s_pred.png"%(file_name),res)
	final_out=cv2.addWeighted(resized_image,1,res,0.6,0)
	cv2.imwrite("%s_pred_overlayed.png"%(file_name),final_out)
	print("")
	print("Finished!")
	print("Wrote image " + "%s_pred.png"%(file_name))

	# USAGE
	# python segment.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --image images/example_01.png

	# load the class label names
	CLASSES = open(classes).read().strip().split("\n")

	# if a colors file was supplied, load it from disk
	if colors:
		COLORS = open(colors).read().strip().split("\n")
		COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
		COLORS = np.array(COLORS, dtype="uint8")
	#print(COLORS)
	legend = np.zeros(((len(CLASSES) * 15) + 15, 300, 3), dtype="uint8")+255

	# loop over the class names + colors
	for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
		# draw the class name + color on the legend
		color = [int(c) for c in color]
		cv2.putText(legend, ' '+className, (0, (i * 15) + 14),
			cv2.FONT_HERSHEY_SIMPLEX, 0.4, tuple(color), 1, cv2.LINE_AA)
		cv2.rectangle(legend, (100, (i * 15)), (300, (i * 15) + 15),tuple(color), -1)

	leg=cv2.cvtColor(np.uint8(legend), cv2.COLOR_RGB2BGR)
	cv2.imwrite("legend.png",leg)
	
	return [img,file_name + '_pred_overlayed.png','legend.png']
'''
	# show the input and output images
	# cv2.namedWindow('legend', cv2.WINDOW_NORMAL)
	cv2.imshow("Legend", legend)
	cv2.imshow("seg_image",res)
	cv2.imshow("overlayed",final_out)
	cv2.imshow("input_image",loaded_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
cv2.imshow("Legend", legend)
'''



