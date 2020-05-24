<h2 align="center" > 
    Face Recognition Using DeepFace<br/>
</h2>
<h3 align="center" > 
    CMPE – 258 Deep Learning<br/>
    Swayam Swaroop Mishra<br/>
    ID - 013725595<br/>
</h3>

Read Me –
1.  Files –
    1.  dataset – (contains images for training)
        1.  swayam – 6 images
        2.	bill gates – 8 images
        3.	steve jobs – 7 images
        4.	unknown – 5 images
    2.	image_test – contains images of people for training.
  3.	video_test – contains images of people for testing.
  4.	face_detection_model
    1.	deploy.protxt
    2.	res10_300x300_ssd_iter_140000.caffemodel
    3.	shape_predictor_68_face_landmarks.dat
  5.	output – 
    1.	dataset_roi – extracted faces
      1.	swayam – 6 images
      2.	bill gates – 8 images
      3.	steve jobs – 7 images
      4.	unknown – 5 images
    2.	embeddings.pickle
    3.	le.pickle
    4.	recognizer.pickle
  6.	pyimagesearch
    1.	__init__.py
    2.	__pycache__
    3.	centroidtracker.pyc
    4.	__init__.pyc
    5.	centroidtracker.py
  7.	openface_nn4.small2.v1.t7
  8.	align.py
  9.	faceTracker1.py
  10.	faceTracker2.py
  11.	deepFaceTrained1.py
  12.	deepFaceTrained2.py
  13.	deepFaceTrained2_video.py
  14.	deepFaceTrained2_webcam.py
2.	Requirements –
  1.	Web Camera
  2.	Python Version – 3.7.6
  3.	Numpy
  4.	OpenCv
  5.	Pyimagesearch
  6.	Argparse
  7.	Imutils
  8.	Pickle
  9.	Scikit Learn
  10.	dlib
3.	Steps to run digit recognition –
  1.	Download all the files into the same directory.
  2.	In the terminal run the following scripts from top to down:
    1.	 If having more than two python version installed in your system (Mac Users) – 
      1.	Face Tracker 1 – Extract faces and align them – python3 faceTracker1.py
      2.	Face Tracker 2 – Normalize the face and embed it to a pickle file – python3 faceTracker2.py
      3.	Deep Face Trained 1 – Train the model – python3 deepFaceTrained1.py
      4.	Deep Face Trained 2 – 
        1.	Test on Still Images – python3 deepFaceTrained2.py –i image_test/image_file_name
        2.	Test on Videos – python3 deepFaceTrained2_video.py -v video_test/video_file_name
        3.	Test on Live Stream or Web Camer – python3 deepFaceTrained2_webcam.py
        
    2. Other Users – 
      1.	Face Tracker 1 – Extract faces and align them – python faceTracker1.py
      2.	Face Tracker 2 – Normalize the face and embed it to a pickle file – python faceTracker2.py
      3.	Deep Face Trained 1 – Train the model – python deepFaceTrained1.py
      4.	Deep Face Trained 2 – 
        1.	Test on Still Images – python deepFaceTrained2.py –i image_test/image_file_name
        2.	Test on Videos – python deepFaceTrained2_video.py -v video_test/video_file_name
        3.	Test on Live Stream or Web Camer – python deepFaceTrained2_webcam.py

References –
1.	Harry Li OpenCV.
2.	Martin Krasser Face Recognition.
3.	Object Tracking - Simple object tracking with OpenCV
4.	Face Tracking - OpenCV Face Recognition
