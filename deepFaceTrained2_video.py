from pyimagesearch.centroidtracker import CentroidTracker
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
 	help="path to video for face detection")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

detector_path = 'face_detection_model'
embedding_model = 'openface_nn4.small2.v1.t7'
recognizer_model = 'output/recognizer.pickle'
path_labelEncoder = 'output/le.pickle'

if not os.path.exists(detector_path):
    print ("ERROR: No OpenCV's deep learning face detector found.")
    exit(-1)
elif not os.path.exists(embedding_model):
    print ("ERROR: No OpenCV's deep learning face embedding model found.")
    exit(-1)
elif not os.path.exists(recognizer_model):
    print ("ERROR: No trained model to recognize faces found.")
    exit(-1)
elif not os.path.exists(path_labelEncoder):
    print ("ERROR: No label encoder found.")
    exit(-1)

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([detector_path, "deploy.prototxt"])
modelPath = os.path.sep.join([detector_path,
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedding_model)

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(recognizer_model, "rb").read())
le = pickle.loads(open(path_labelEncoder, "rb").read())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()

# initialize the video stream
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
time.sleep(2.0)
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)


# loop over frames from the video file stream
while vs.isOpened():
	# grab the frame from the threaded video stream
    ret, frame = vs.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=400)

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    rects = []

	# construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

	# loop over the detections
    for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
        confidence = detections[0, 0, i, 2]

		# filter out weak detections
        if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            rects.append(box.astype("int"))
			# extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

			# perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

			# draw the bounding box of the face along with the
			# associated probability
            if name == "unknown":
                text = "{}".format(name)
            else:
                text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 255), -1)

	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


# do a bit of cleanup
cv2.destroyAllWindows()
