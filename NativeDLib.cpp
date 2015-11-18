#include "NativeDLib.h"
#include <fstream> 
#include <string> 


NativeDLib::NativeDLib()
{
	detector = dlib::get_frontal_face_detector();
}

NativeDLib::NativeDLib(std::string faceModelFileName, std::string shapePredictorFileName)
{
	try{
		loadMeanPoints(faceModelFileName);
		loadShapePredictor(shapePredictorFileName);
	}
	catch (const std::invalid_argument& ia) 
	{
		std::cerr << "Invalid argument: " << ia.what() << '\n';
	}
	catch (std::ifstream::failure e) 
	{
    	std::cerr << "Exception opening/reading/closing file\n";
	}
}

void NativeDLib::loadShapePredictor(std::string shapePredictorFileName)
{
	dlib::deserialize(shapePredictorFileName) >> shapePredictor;
}


void NativeDLib::loadMeanPoints(std::string faceModelFileName)
{
	std::ifstream inputFaceModelFile(faceModelFileName);
	checkOpenedFile(inputFaceModelFile);
	std::string line; 
	while(getline(inputFaceModelFile, line)) 
	{
		Point tempPoint = getDoubleFromCSVLine(line);
		meanAveragePoints.push_back(tempPoint);
	}
}

void NativeDLib::checkOpenedFile(std::ifstream &inFile)
{
	if(!inFile.good())
		throw std::invalid_argument("Unable to process file.");;
}

int NativeDLib::getNumberOfPoints()
{
	return meanAveragePoints.size();
}

Point NativeDLib::getDoubleFromCSVLine(std::string line)
{
	double x,y;
	std::size_t commaPos = line.find(',');
	std::string firstPart = line.substr(0, commaPos);
	x = std::stod(firstPart);
	std::string secondPart = line.substr(commaPos+1);
	y = std::stod(secondPart);
	Point res(x,y);
	return res;
}

void NativeDLib::printFaceMeanList()
{
	for(Point curPoint: meanAveragePoints){
		std::cout<<curPoint.x<<" "<<curPoint.y<<std::endl;
	}
}

BoundingBox NativeDLib::getLargestFaceBoundingBox(cv::Mat cvimg)
{
	loadShapePredictor("shape_predictor_68_face_landmarks.dat");
	
	cvDlibImage img(cvimg);
	std::vector<BoundingBox> faces = detector(img);
	BoundingBox largestFace; 
	int largestArea = 0;
	for(BoundingBox face:faces){
		int currentArea = face.width()*face.height();
		if(currentArea>largestArea)
		{
			largestArea = currentArea;
			largestFace = face;
		}
	}
	return largestFace;
}

BoundingBox NativeDLib::getLargestFaceBoundingBox(Image &img)
{
	std::vector<BoundingBox> faces = detector(img);
	BoundingBox largestFace; 
	int largestArea = 0;
	for(BoundingBox face:faces){
		int currentArea = face.width()*face.height();
		if(currentArea>largestArea)
		{
			largestArea = currentArea;
			largestFace = face;
		}
	}
	return largestFace;
}

cv::Mat NativeDLib::alignImg(std::string method, Image &img, BoundingBox bb,  std::string imgName)
{
	PointList alignPoints = align(img, bb);
	PointList meanAlignPoints = transformPoints(meanAveragePoints, bb);
	int left=meanAlignPoints[0].x(), top=meanAlignPoints[0].y(), 
		right=meanAlignPoints[0].x(), bottom=meanAlignPoints[0].y();
	for(int i=0; i<meanAlignPoints.size(); i++)
	{
		left = std::min((long)left, meanAlignPoints[i].x());
		top = std::min((long)top, meanAlignPoints[i].y());
		right = std::max((long)right, meanAlignPoints[i].y());
		bottom = std::max(long(bottom), meanAlignPoints[i].y());
	}
	dlib::rectangle tightBb(left, top, right, bottom);
	int ss[3] ={39, 42, 57}; 
	
	cv::Point2f alignPointsSS[3];
	cv::Point2f meanAlignPointsSS[3]; 
	for(int i=0; i<3; i++){
		alignPointsSS[i].x = alignPoints[ss[i]].x(); 
		alignPointsSS[i].y = alignPoints[ss[i]].y();
		
		meanAlignPointsSS[i].x = meanAlignPoints[ss[i]].x();
		meanAlignPointsSS[i].y = meanAlignPoints[ss[i]].y();
	}
	cv::Mat H = cv::getAffineTransform(alignPointsSS, meanAlignPointsSS);
	
	cv::Mat cvImg = dlibImgtoCV(img);
	cv::Mat warpedImg = cv::Mat::zeros( cvImg.rows, cvImg.cols, cvImg.type());
	
	cv::warpAffine(cvImg, warpedImg, H, warpedImg.size());
	
	BoundingBox wBb = getLargestFaceBoundingBox(warpedImg);
	if(wBb.width()<=0 || wBb.height()<=0)
		throw std::invalid_argument("Error with bounding box.");
	PointList wAlignPoints = align(warpedImg, wBb);
	PointList wMeanAlignPoints = transformPoints(meanAveragePoints, wBb);
	
	if(warpedImg.channels()!=3)
		throw std::invalid_argument("Image does not have 3 channels.");
		
	left=wAlignPoints[0].x(), top=wAlignPoints[0].y(), 
		right=wAlignPoints[0].x(), bottom=wAlignPoints[0].y();
	for(int i=0; i<wAlignPoints.size(); i++)
	{
		left = std::min((long)left, wAlignPoints[i].x());
		top = std::min((long)top, wAlignPoints[i].y());
		right = std::max(long(right), wAlignPoints[i].y());
		bottom = std::max(long(bottom), wAlignPoints[i].y());
	}
	
	int w = warpedImg.size[0], h = warpedImg.size[1];
	
	if (!(0 <= left && left <= w && 0 <= right && right <= w &&
		 0 <= bottom && bottom <= h && 0 <= top && top <= h))
        throw std::invalid_argument("Warning: Unable to align and crop to the "
              "face's bounding box.");

	cv::Rect wrect(wBb.left(), wBb.top(), wBb.width(), wBb.height());
	cv::rectangle(warpedImg, wrect,  cv::Scalar(0, 0, 255));
    return warpedImg;
}

PointList NativeDLib::align(Image &img, BoundingBox faceBB)
{
	PointList res;
	dlib::full_object_detection shape = shapePredictor(img, faceBB);
	for(int i=0; i<shape.num_parts(); i++)
	{
		res.push_back(shape.part(i));	
	}
	return res;
}

PointList NativeDLib::align(cv::Mat cvImg, BoundingBox faceBB)
{
	PointList res;
	cvDlibImage cvdlibimg(cvImg);
	dlib::full_object_detection shape = shapePredictor(cvdlibimg, faceBB);
	for(int i=0; i<shape.num_parts(); i++)
	{
		res.push_back(shape.part(i));	
	}
	return res;
}

PointList NativeDLib::transformPoints(AvgPointList points, BoundingBox faceBB)
{
	PointList res;
	for(int i=0; i<points.size(); i++)
	{
		double x = points[i].x, y = points[i].y;
		dlib::point transformedP(int((x * faceBB.width()) + faceBB.left()), 
					int((y * faceBB.height()) + faceBB.top()));
		res.push_back(transformedP);			
	}	
	return res;
}


cv::Mat NativeDLib::dlibImgtoCV(Image &img)
{
	return  dlib::toMat(img);
}
//////////////////////////////////////////////////////////////
//////////////////////// TESTS  //////////////////////////////
//////////////////////////////////////////////////////////////

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
		Image img;
		std::string imgName = "sample.jpg";
		dlib::load_image(img, imgName);
		BoundingBox face = align.getLargestFaceBoundingBox(img);
		align.loadShapePredictor("shape_predictor_68_face_landmarks.dat");
		bool notZero = (face.width()>0);
		CHECK_EQUAL(notZero, 1);
    }	
}

int main(int argc, const char *argv[] )
{
	if(argc>1){
		for(int i=1; i<argc; i++){
			NativeDLib align;
			Image img;
			std::string imgName = argv[i];
			align.loadMeanPoints("models/dlib/mean.csv");
			align.loadShapePredictor("shape_predictor_68_face_landmarks.dat");
			dlib::load_image(img, imgName);
			BoundingBox bb = align.getLargestFaceBoundingBox(img);
			cv::Mat alignedFace = align.alignImg("affine",img, bb, imgName);
			cv::imwrite("aligned_"+imgName,  alignedFace);
		}
	}
	return UnitTest::RunAllTests();
}