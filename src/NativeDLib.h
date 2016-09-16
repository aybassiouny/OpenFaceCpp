#pragma once

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include "opencv2/opencv.hpp"

namespace OpenFaceCpp
{
    using PointList = std::vector<dlib::point>;
    using DlibImage = dlib::array2d<dlib::bgr_pixel>;
    using DlibOpenCVImage = dlib::cv_image<dlib::bgr_pixel>;
    using AvgPointList = std::vector<cv::Point2d>;

    class NativeDLib
    {
    public:
        NativeDLib(const std::string& configFileName);
        
        dlib::rectangle GetLargestFaceBoundingBox(const DlibImage &img);
        dlib::rectangle GetLargestFaceBoundingBox(const cv::Mat& img);

        cv::Mat AlignImg(int imgDim, DlibImage &img, const dlib::rectangle& bb);

    private:
        void PrintFaceMeanList();
        void LoadMeanPoints(const std::string& faceModelName);

        PointList TransformPoints(const AvgPointList& points, const dlib::rectangle& faceBB);
        PointList Align(const cv::Mat& cvImg, const dlib::rectangle& faceBB);

        DlibImage m_img;
        dlib::frontal_face_detector m_detector;
        dlib::shape_predictor m_shapePredictor;
        AvgPointList m_meanAveragePoints;
    };

}