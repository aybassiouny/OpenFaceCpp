#include "NativeDLib.h"
#include "TorchWrap.h"

#include "opencv2/opencv.hpp"
#include <dlib/image_io.h>

int main(int argc, const char *argv[] )
{
    if(argc<3)
    {
        std::cout<<"Please use: OpenFaceCpp <config_file_name> <image_name>"<<std::endl;
        return 0;
    }

    
    OpenFaceCpp::TorchWrap tw(argv[1]);

    std::vector<float> result;
    tw.ForwardImage(argv[2], result);

    std::cout << "Received following output: " << std::endl;
    for(const auto value : result)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}
