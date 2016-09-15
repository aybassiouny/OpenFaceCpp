#include "NativeDLib.h"

#include <tinyxml2.h>

#include <chrono>

using namespace OpenFaceCpp;

using namespace std::chrono;

NativeDLib::NativeDLib(const std::string& configFileName)
{
    try
    {
        tinyxml2::XMLDocument doc;
        doc.LoadFile(configFileName.c_str());
        std::string faceModelFileName = doc.FirstChildElement("FaceModelFileName")->GetText();
        std::string shapePredictorFileName = doc.FirstChildElement("ShapePredictorFileName")->GetText();

        LoadMeanPoints(faceModelFileName);
        dlib::deserialize(shapePredictorFileName) >> m_shapePredictor;

        std::istringstream sin(dlib::get_serialized_frontal_faces());
        dlib::deserialize(m_detector, sin);
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

void NativeDLib::LoadMeanPoints(const std::string& faceModelFileName)
{
    std::ifstream inputFaceModelFile(faceModelFileName);
    CheckOpenedFile(inputFaceModelFile);
    std::string line; 
    while(std::getline(inputFaceModelFile, line)) 
    {
        cv::Point2d tempPoint;
        GetDoubleFromCSVLine(line, tempPoint);
        m_meanAveragePoints.push_back(std::move(tempPoint));
    }
}

void NativeDLib::CheckOpenedFile(const std::ifstream& inFile)
{
    if(!inFile.good())
        throw std::invalid_argument("Unable to process file.");;
}

int NativeDLib::GetNumberOfPoints()
{
    return m_meanAveragePoints.size();
}

void NativeDLib::GetDoubleFromCSVLine(const std::string& line, cv::Point2d& outputPoint )
{    
    std::size_t commaPos = line.find(',');
    std::string firstPart = line.substr(0, commaPos);
    outputPoint.x = std::stod(firstPart);
    std::string secondPart = line.substr(commaPos+1);
    outputPoint.y = std::stod(secondPart);
}

void NativeDLib::PrintFaceMeanList()
{
    for (auto curPoint: m_meanAveragePoints)
    {
        std::cout<<curPoint.x<<" "<<curPoint.y<<std::endl;
    }
}

dlib::rectangle NativeDLib::GetLargestFaceBoundingBox(const cv::Mat& cvimg)
{
    DlibOpenCVImage img(cvimg);
    std::vector<dlib::rectangle> faces = m_detector(img);
    dlib::rectangle largestFace;
    int largestArea = 0;
    for(dlib::rectangle face:faces){
        int currentArea = face.width()*face.height();
        if(currentArea>largestArea)
        {
            largestArea = currentArea;
            largestFace = face;
        }
    }
    return largestFace;
}

dlib::rectangle NativeDLib::GetLargestFaceBoundingBox(const DlibImage& img)
{
    std::vector<dlib::rectangle> faces = m_detector(img);
    dlib::rectangle largestFace;
    int largestArea = 0;
    for(const auto& face:faces){
        int currentArea = face.width()*face.height();
        if(currentArea>largestArea)
        {
            largestArea = currentArea;
            largestFace = face;
        }
    }
    return largestFace;
}

cv::Mat NativeDLib::AlignImg(int imgDim, DlibImage &img, const dlib::rectangle& bb)
{
    auto dlibImb = DlibImgtoCV(img);
    PointList alignPoints = Align(dlibImb, bb);
    PointList meanAlignPoints = TransformPoints(m_meanAveragePoints, bb);

    if (meanAlignPoints.size() < 1)
    {
        throw std::range_error("Mean align points Error. Did you laod Face Model?");
    }

    int left=meanAlignPoints[0].x(), top=meanAlignPoints[0].y(), 
        right=meanAlignPoints[0].x(), bottom=meanAlignPoints[0].y();
    
    for(std::size_t i=0; i<meanAlignPoints.size(); i++)
    {
        left = std::min<long>(left, meanAlignPoints[i].x());
        top = std::min<long>(top, meanAlignPoints[i].y());
        right = std::max<long>(right, meanAlignPoints[i].y());
        bottom = std::max<long>(bottom, meanAlignPoints[i].y());
    }

    dlib::rectangle tightBb(left, top, right, bottom);
    int ss[3] ={39, 42, 57}; 
    cv::Point2f alignPointsSS[3];
    cv::Point2f meanAlignPointsSS[3]; 

    for(std::size_t i=0; i<3; i++)
    {
        alignPointsSS[i].x = alignPoints[ss[i]].x(); 
        alignPointsSS[i].y = alignPoints[ss[i]].y();
        
        meanAlignPointsSS[i].x = meanAlignPoints[ss[i]].x();
        meanAlignPointsSS[i].y = meanAlignPoints[ss[i]].y();
    }

    cv::Mat H = cv::getAffineTransform(alignPointsSS, meanAlignPointsSS);
    
    cv::Mat cvImg = DlibImgtoCV(img);
    cv::Mat warpedImg = cv::Mat::zeros( cvImg.rows, cvImg.cols, cvImg.type());
    cv::warpAffine(cvImg, warpedImg, H, warpedImg.size());
    dlib::rectangle wBb = GetLargestFaceBoundingBox(warpedImg);
    
    if (wBb.width() <= 0 || wBb.height() <= 0)
    {
        throw std::invalid_argument("Error with bounding box.");
    }

    PointList wAlignPoints = Align(warpedImg, wBb);
    PointList wMeanAlignPoints = TransformPoints(m_meanAveragePoints, wBb);
    
    if(warpedImg.channels()!=3)
        throw std::invalid_argument("Image does not have 3 channels.");
        
    left=wAlignPoints[0].x(), top=wAlignPoints[0].y(), 
        right=wAlignPoints[0].x(), bottom=wAlignPoints[0].y();

    for(std::size_t i=0; i<wAlignPoints.size(); i++)
    {
        left = std::min<long>(left, wAlignPoints[i].x());
        top = std::min<long>(top, wAlignPoints[i].y());
        right = std::max<long>(right, wAlignPoints[i].y());
        bottom = std::max<long>(bottom, wAlignPoints[i].y());
    }

    int w = warpedImg.size[0], h = warpedImg.size[1];
    
    if (!(0 <= left && left <= w && 0 <= right && right <= w &&
        0 <= bottom && bottom <= h && 0 <= top && top <= h))
    {
        throw std::invalid_argument("Warning: Unable to align and crop to the "
            "face's bounding box.");
    }

    cv::Rect wrect(wBb.left(), wBb.top(), wBb.width(), wBb.height());
    cv::rectangle(warpedImg, wrect,  cv::Scalar(0, 0, 255));
    cv::resize(warpedImg, warpedImg, cv::Size(imgDim, imgDim));
    
    return warpedImg;
}

PointList NativeDLib::Align(const cv::Mat& cvImg, const dlib::rectangle& faceBB)
{
    PointList res;
    DlibOpenCVImage cvdlibimg(cvImg);
    dlib::full_object_detection shape = m_shapePredictor(cvdlibimg, faceBB);
    for(std::size_t i=0; i<shape.num_parts(); i++)
    {
        res.push_back(shape.part(i));	
    }

    return res;
}

PointList NativeDLib::TransformPoints(const AvgPointList& points, const dlib::rectangle& faceBB)
{
    PointList res;
    for(std::size_t i=0; i<points.size(); i++)
    {
        double x = points[i].x, y = points[i].y;
        dlib::point transformedP(int((x * faceBB.width()) + faceBB.left()), 
                    int((y * faceBB.height()) + faceBB.top()));
        res.push_back(transformedP);			
    }

    return res;
}


cv::Mat NativeDLib::DlibImgtoCV(DlibImage& img)
{
    return  dlib::toMat(img);
}