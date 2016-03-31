rm OpenFaceCpp
cd build
cmake  -DCMAKE_BUILD_TYPE=Release -DUSE_AVX_INSTRUCTIONS=ON ..
make -j12
mv OpenFaceCpp ../
cd ..
./OpenFaceCpp
