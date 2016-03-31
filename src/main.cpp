#include "NativeDLib.h"
#include "UnitTest++/UnitTest++.h"
#include "opencv2/opencv.hpp"
#include "TorchWrap.h"
using namespace cv;

SUITE(NativeDlibTest)
{
    class NativeDlibFixture
    {
    public:
        NativeDLib align;
    };

    TEST_FIXTURE(NativeDlibFixture, openFile)
    {
		align.loadMeanPoints("models/dlib/mean.csv");
		int numPoints = align.getNumberOfPoints();
		CHECK_EQUAL(numPoints, 68);
    }

    TEST_FIXTURE(NativeDlibFixture, invalidFile)
    {
        CHECK_THROW(align.loadMeanPoints("invalid.csv"), std::invalid_argument);
    }
	
	
	TEST_FIXTURE(NativeDlibFixture, shapePredictor)
    {
		align.loadMeanPoints("models/dlib/mean.csv");
		align.loadShapePredictor("shape_predictor_68_face_landmarks.dat");
		bool notZero = (align.shapePredictor.num_parts()>0);
		CHECK_EQUAL(notZero, 1);
    }
	
	TEST_FIXTURE(NativeDlibFixture, getlargestBB)
    {
		NativeDLib align("models/dlib/mean.csv", "shape_predictor_68_face_landmarks.dat");
        Image img;
        std::string imgName = "sample.jpg";
        dlib::load_image(img, imgName);
        BoundingBox face = align.getLargestFaceBoundingBox(img);
        bool notZero = (face.width()>0);
        std::cout<<face.width()<<" "<<face.height()<<std::endl;
        cv::Mat alignedFace = align.alignImg("affine", 96, img, face, "blabla");
        cv::imwrite("output.jpg", alignedFace);
		
		CHECK_EQUAL(notZero, 1);
    }	
}

int main(int argc, const char *argv[] )
{
#ifdef NOTESTS
	VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!cap.open(0))
        return 0;
	NativeDLib align;
	//align.loadMeanPoints("models/dlib/mean.csv");
	align.loadShapePredictor("shape_predictor_68_face_landmarks.dat");
    while(true)
    {
          Mat frame;
          cap >> frame;
          if( frame.empty() ) break; // end of video stream
		  BoundingBox bb = align.getLargestFaceBoundingBox(frame);
		  rectangle(frame, cv::Point(bb.left(), bb.top()), 
			  cv::Point(bb.right(), bb.bottom()), Scalar(1,0,0) );
          imshow("Webcam", frame);
          if( waitKey(1) == 27 ) break; // stop capturing by pressing ESC 
    }
	return 0;
#else
    if(argc<2)
    {
        std::cout<<"Please provide input image"<<std::endl;
        return 0;
    }
    NativeDLib align("models/dlib/mean.csv", "shape_predictor_68_face_landmarks.dat");
    Image img;
    std::string imgName = argv[1];
    dlib::load_image(img, imgName);
    BoundingBox face = align.getLargestFaceBoundingBox(img);
    cv::Mat alignedFace = align.alignImg("affine", 96, img, face, "blabla");
    cv::imwrite("temp_aligned.jpg", alignedFace);
    TorchWrap tw;
    auto ans = tw.forwardImage("temp_aligned.jpg");
    //return UnitTest::RunAllTests();
#endif
}