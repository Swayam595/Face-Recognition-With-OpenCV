from imutils import paths
import imutils
import pickle
import cv2
import os

roi_path = 'output/dataset_roi'
embedding_path = 'output/embeddings.pickle'
embedding_model = 'openface_nn4.small2.v1.t7'

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedding_model)

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(roi_path))


# initialize our lists of extracted facial embeddings and
# corresponding people names
knownEmbeddings = []
knownNames = []

# initialize the total number of faces processed
total = 0

# loop over the image paths
# loop over the image paths
for i, imagePath in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePath)))
    name = imagePath.split(os.path.sep)[-2]

    # load the image, resize it to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
    face = cv2.imread(imagePath)

    # construct a blob for the face ROI, then pass the blob
	# through our face embedding model to obtain the 128-d
	# quantification of the face
    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
            (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()

	# add the name of the person + corresponding face
	# embedding to their respective lists
    knownNames.append(name)
    knownEmbeddings.append(vec.flatten())
    total += 1

# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(embedding_path, "wb")
f.write(pickle.dumps(data))
f.close()
