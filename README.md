# OpenFaceCpp
C++ implementation for [OpenFace](https://github.com/cmusatyalab/openface) library by CMU. This is still a work in progress. 

We utilize all their dependencies: dlib, torch and OpenCV, besides a minimal testing library: UnitTest++. All is inlucded except for OpenCV that you need to have pre-installed on your system. 

# System support
Up to this point (check below) the system if up and running over Ubuntu, CentOS and Windows. It's expected for windows to fall back later on due to the torch dependency - but someone maybe could make their unofficial release work. 

# Install
For Linux, run install.sh. For the resulting executable to run, you need to download models first from: and place it in same folder with executable. 

In order to run the executable you will need to download [dlib's model](http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2) file and place it in the root directory. 

For Windows, first cmake Lib/Unittest++, build it, then cmake the root directory. 

# Todo list
- Face detection and alignment ✔
- Euclidean face representation ✔ 
- Integration into Face Recognitino sample ✔
- sync with latest OpenFace V0.2.0
