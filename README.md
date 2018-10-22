# OpenFaceCpp
C++ implementation for [OpenFace](https://github.com/cmusatyalab/openface) library by CMU.

Although there are many dependencies, the repo tries to minimize manual setup steps by using hunter packaging. Currently the only package that needs to be manually setup is torch.

# Torch setup
We use torch to run our models. I had the following setup: 
- Torch windows binaries from [here](https://github.com/hiili/WindowsTorch/releases/tag/64-bit_LuaJIT-2-1b2_2017-07)
- `dpnn` required package by running [this hack](https://github.com/Element-Research/dpnn/issues/91#issuecomment-301303536) 
- `libjpeg` on windows, then re-ran `luarocks install image` in order to make sure it is linked to libjpeg (otherwise torch cannot deal with jpegs)
- Ascii models from [here ](https://github.com/cmusatyalab/openface/issues/42#issuecomment-198406159) since binary models won't work. 
- Good luck :)  

# Install
This is a standard cmake project. Create a build directory from root, then `cmake ..`, followed by `cmake --build .` should do the trick. 

OpenFaceCpp currently takes an an input an xml config file. A sample is in `src/OpenFaceConfig.xml`. You will see a couple models are needed, they can be found in the [original repo](https://github.com/cmusatyalab/openface/blob/master/models/get-models.sh). 

E.g. OpenFaceConfig.xml expects:
- shape_predictor_68_face_landmarks.dat: it's possible to download it by running in PS: 
`wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O dlib/shape_predictor_68_face_landmarks.dat.bz2`
- nn4.small2.v1: get it (and other models) from [here](http://cmusatyalab.github.io/openface/models-and-accuracies/)
