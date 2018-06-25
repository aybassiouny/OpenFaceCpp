#include "NativeDLib.h"
#include "TorchWrap.h"

#include <dlib/image_io.h>

int main(int argc, const char *argv[] )
{
    if(argc<3)
    {
        std::cout<<"Please use: OpenFaceCpp <config_file_name> <image_name>"<<std::endl;
        return 0;
    }

    
    std::vector<float> result;

    try
    {
        OpenFaceCpp::TorchWrap tw(argv[1]);
        tw.ForwardImage(argv[2], result);
    }
    catch(std::exception e)
    {
        std::cout << "Failed to process image. Stopping excution."<< std::endl;
        return 0;
    }
    

    std::cout << "Received result feature vector of size : " <<result.size()<< std::endl;
}
