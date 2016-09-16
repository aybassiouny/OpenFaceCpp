# OpenFaceCpp
C++ implementation for [OpenFace](https://github.com/cmusatyalab/openface) library by CMU.  

Although there are many depedencies, the repo tries to minimize manual setup steps by using hunter packaging. Currently the only package needed to get going is torch.

# Install
A standard cmake project. Create a build directory from root, then `cmake ..`, followed by `cmake --build .` should do the trick. 

OpenFaceCpp currently takes an an input an xml config file. A sample is in `src/OpenFaceConfig.xml`. You will see a couple models are needed, they can be found in the [original repo](https://github.com/cmusatyalab/openface/blob/master/models/get-models.sh). 

# What's next? 
I am currently working on figuring out a way to do without the lua stuff. That would be sweet. 
