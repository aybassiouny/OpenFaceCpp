#include "opencv2/opencv.hpp"
#include "NativeDLib.h"
#include "TorchWrap.h"

using namespace cv;
namespace bp = ::boost::process; 
using std::cout;
using std::endl;

NativeDLib align;

vector<double>  getImgRep(string imgName, TorchWrap tw, bp::child& c){
	cv::Mat src = cv::imread(imgName,1); 
	Image img;
	dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(src));
    BoundingBox face = align.getLargestFaceBoundingBox(img);
    Mat alignedFace = align.alignImg("affine", 96, img, face, "blabla");
    cv::imwrite("temp_aligned.jpg", alignedFace);
    auto ans = tw.forwardImage("temp_aligned.jpg", std::forward<decltype(c)>(c));
    return ans;
}

int main(int argc, const char *argv[] )
{
    if(argc<2)
    {
        std::cout<<"Please provide input image"<<std::endl;
        return 0;
    }
	
	std::string imgName = argv[1];
    align.init("models/dlib/mean.csv", "shape_predictor_68_face_landmarks.dat");
    TorchWrap tw("models/openface/nn4.small2.v1", 96, "representation.txt");
	bp::child c = tw.initChild();
	getImgRep(imgName, tw, c);
    //return UnitTest::RunAllTests();
}
