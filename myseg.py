# USAGE
# python segment.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --image images/example_01.png

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--classes", required=True,
	help="path to .txt file containing class labels")
ap.add_argument("-l", "--colors", type=str,
	help="path to .txt file containing colors for labels")
args = vars(ap.parse_args())

# load the class label names
CLASSES = open(args["classes"]).read().strip().split("\n")

# if a colors file was supplied, load it from disk
if args["colors"]:
	COLORS = open(args["colors"]).read().strip().split("\n")
	COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
	COLORS = np.array(COLORS, dtype="uint8")
print(COLORS)
legend = np.zeros(((len(CLASSES) * 15) + 15, 300, 3), dtype="uint8")+255

# loop over the class names + colors
for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
	# draw the class name + color on the legend
	color = [int(c) for c in color]
	cv2.putText(legend, ' '+className, (0, (i * 15) + 14),
		cv2.FONT_HERSHEY_SIMPLEX, 0.4, tuple(color), 1, cv2.LINE_AA)
	cv2.rectangle(legend, (100, (i * 15)), (300, (i * 15) + 15),tuple(color), -1)




# show the input and output images
# cv2.namedWindow('legend', cv2.WINDOW_NORMAL)
cv2.imshow("Legend", legend)
cv2.imwrite("legendtry.png",legend)
cv2.waitKey(0)
cv2.destroyAllWindows()
