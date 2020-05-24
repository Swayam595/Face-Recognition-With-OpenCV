import numpy as np
from imutils import paths
from align import AlignDlib
import argparse
import imutils
import cv2
import os


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

dataset_path = 'dataset'
roi_path = 'output/dataset_roi'
detector_path = 'face_detection_model'

if not os.path.exists(dataset_path):
    print ("ERROR: No dataset directory found. Create a directory dataset and store images of people in it.")
    exit(-1)

if not os.path.exists(roi_path):
    os.mkdir(roi_path)

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([detector_path, "deploy.prototxt"])
modelPath = os.path.sep.join([detector_path,
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(dataset_path))

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
print("[INFO] loading face detector and face aligner...")
predictor_path = os.path.sep.join([detector_path, "shape_predictor_68_face_landmarks.dat"])
alignment = AlignDlib(predictor_path)

# initialize the total number of faces processed
total = 0

# loop over the image paths
for i, imagePath in enumerate(imagePaths):
    # extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1, len(imagePath)))
	name = imagePath.split(os.path.sep)[-2]
	image_name = imagePath.split(os.path.sep)[-1]
    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	(h, w) = image.shape[:2]

    # construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# ensure that the detection with the largest probability also
		# means our minimum probability test (thus helping filter out
		# weak detections)
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI and grab the ROI dimensions
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# Detect face and return bounding box
			bb = alignment.getLargestFaceBoundingBox(image)

			# Transform image using specified face landmark indices and crop image to 96x96
			faceAligned = alignment.align(96, image, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

			dataset_roi_path = '%s/%s'%(roi_path, name)
			if not os.path.exists(dataset_roi_path):
				os.mkdir(dataset_roi_path)

			saved_roi = '%s/%s'%(dataset_roi_path, image_name)
			cv2.imwrite(saved_roi, faceAligned)

			total += 1

print ('Total ROIs extracted: %s'%total)
