#include "NativeDLib.h"
#include "TorchWrap.h"

#include "opencv2/opencv.hpp"
#include <dlib/image_io.h>


using namespace OpenFaceCpp;
using std::cout;
using std::endl;

std::vector<double>  GetImgRep(NativeDLib& align, const std::string& imgName, TorchWrap& tw, const boost::process::child& c)
{
    cv::Mat src = cv::imread(imgName,1); 
    DlibImage img;
    dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(src));
    dlib::rectangle face = align.GetLargestFaceBoundingBox(img);
    cv::Mat alignedFace = align.AlignImg(96, img, face, "blabla");

    std::string inputImgName = "temp_aligned.jpg";
    cv::imwrite(inputImgName, alignedFace);
    auto ans = tw.ForwardImage(inputImgName, c);
    return ans;
}

int main(int argc, const char *argv[] )
{
    if(argc<2)
    {
        std::cout<<"Please provide input image"<<std::endl;
        return 0;
    }
    
    NativeDLib align(argv[0]);
    std::string imgName = argv[1];
    
    TorchWrap tw(argv[0]);
    boost::process::child c = tw.initChild();
    GetImgRep(align, imgName, tw, c);
}
