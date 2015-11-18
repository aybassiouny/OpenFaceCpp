#include "UnitTest++/UnitTest++.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream> 
#include <string> 
#include <algorithm> 

class Point{
public:
	double x;
	double y;
	Point(double i,double j)
	{ 
		x = i; y = j;
	};
};

typedef std::vector<dlib::point> PointList;
typedef dlib::rectangle BoundingBox;
typedef dlib::array2d<dlib::bgr_pixel>  Image;
typedef dlib::cv_image<dlib::bgr_pixel> cvDlibImage;
typedef std::vector<Point> AvgPointList;
class NativeDLib 
{
public: 
	Image img;
	dlib::frontal_face_detector detector;
	dlib::shape_predictor shapePredictor;
	AvgPointList meanAveragePoints;
	
	Point getDoubleFromCSVLine(std::string line);
public: 
	NativeDLib();	
	NativeDLib(std::string faceModelFileName, 
				std::string shapePredictorFileName);
	BoundingBox getLargestFaceBoundingBox(Image &img);
	BoundingBox getLargestFaceBoundingBox(cv::Mat img);
	
	void loadShapePredictor(std::string shapePredictorFileName);
	void printFaceMeanList();
	int getNumberOfPoints();
	void loadMeanPoints(std::string faceModelName);
	void checkOpenedFile(std::ifstream &inFile);
	
	cv::Mat alignImg(std::string method, Image &img, BoundingBox bb, std::string imgName);
	PointList transformPoints(AvgPointList points, BoundingBox faceBB);
	PointList align(Image &img, BoundingBox faceBB);
	PointList align(cv::Mat cvImg, BoundingBox faceBB);
	cv::Mat dlibImgtoCV(Image &img);
};

