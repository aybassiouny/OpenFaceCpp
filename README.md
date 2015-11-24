# OpenFaceCpp
C++ implementation for OpenFace library by CMU https://github.com/cmusatyalab/openface. This is still a work in progress. 

We utilize all their dependencies: dlib, torch and OpenCV, besides a minimal testing library: UnitTest++. All is inlucded except for OpenCV that you need to have pre-installed on your system. 

# System support
Up to this point (check below) the system if up and running over Ubuntu, CentOS and Windows. It's expected for windows to fall back later on due to the torch dependency - but someone maybe could make their unofficial release work. 

# Install
Run install.sh. For the resulting executable to run, you need to download models first from: and place it in same folder with executable. 

# Todo list
- Face detection and alignment ✔
- Euclidean face representation 
- Integration into Face Recognitino sample 
