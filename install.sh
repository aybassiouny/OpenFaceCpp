cd Lib/unittest-cpp/
rm -rf builds
mkdir builds
cd builds
cmake -G "Unix Makefiles" ../
cmake --build ./
cd ../../..
rm -rf build
mkdir build
cd build
cmake ..
make
mv OpenFaceCpp ../
cd ..