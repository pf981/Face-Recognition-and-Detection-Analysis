This project was a university assignment for Computer Vision that analysed computer vision algorithms. This code trains and runs performance tests several face detection and recognition algorithms. The algorithms examined are Haar Cascade Classifiers for face detection and Eigenfaces, Fisherfaces, Local Binary Patters (LBP) and Local Intensity Distribution (LID) for face recognition.

Please see report.pdf for the complete analysis.

CSCI435 Computer Vision
=======================
Project: Facial Recognition
===========================

Testing Environment
-------------------
This code was tested on Ubuntu 12.04 LTS with OpenCV 2.4.2.

How to Run
----------
The code can be compiled an run in the following way:
    $ make
    $ ./faceDetection [image]

Additional Notes
----------------
Please note that I have included the trained XML files in the submission:
 * haarcascade_frontalface_alt2.xml
 * trained_eigen.xml
 * trained_fisher.xml
 * trained_lbp.xml
 * trained_lid.xml

This means that you do not have to run the training functions. It is important that you don't run them, as it assumes that the face_samples folder is with the executable, and I also had to rename a few images.
Similarly, there is no need to run the performance testing functions as the results can be found
within the report.

For the report, please see report.pdf
