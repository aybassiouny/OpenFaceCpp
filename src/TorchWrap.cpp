#include "TorchWrap.h"
#include "tinyxml2.h"

#include <iostream>
#include <string>
#include <chrono>

#include <boost/iostreams/stream.hpp>

using namespace OpenFaceCpp;

using namespace std::chrono;
using namespace boost::process;

namespace 
{
    void StringToVecOfStrings(const std::string& str, std::vector<std::string>& res)
    {
        std::stringstream ss(str);
        std::string substr;
        while(ss>> substr)
        {
            res.push_back(std::move(substr));
        }
    }
}

TorchWrap::Executer::Executer(const std::string& modelPath, int imgDim)
    :
    m_appToChild{ create_pipe() },
    m_childToApp{ create_pipe() }
{
    m_childSink.open(m_appToChild.sink, boost::iostreams::close_handle);
    m_childSource.open(m_childToApp.source, boost::iostreams::close_handle);

    const std::string c_exec = "th_emulator.exe openface_server.lua -model " + modelPath + " -imgDim " + std::to_string(imgDim);
    std::vector<std::string > commandWithArguments;
    StringToVecOfStrings(c_exec, commandWithArguments);

    std::cout << "Launching " << c_exec << std::endl;
    m_child = std::make_unique<boost::process::child>(
        execute(initializers::set_args(commandWithArguments),
            initializers::bind_stdout(m_childSink),
            initializers::bind_stdin(m_childSource)));

    m_appSource.open(m_appToChild.source, boost::iostreams::close_handle);
    m_childSink.open(m_childToApp.sink, boost::iostreams::close_handle);
}

bool TorchWrap::Executer::SendMsg(const std::string& msg)
{
    boost::iostreams::stream<boost::iostreams::file_descriptor_sink> appOutChildIn(m_childSink);
    appOutChildIn << msg << std::endl;
    return true;
}

bool TorchWrap::Executer::ReceiveMsg(std::string& msg)
{
    std::cout << "listening ..." << std::endl;
    boost::iostreams::stream<boost::iostreams::file_descriptor_source> childOutAppin(m_appSource);
    std::getline(childOutAppin, msg);
    std::cout << "Received: " << msg << std::endl;
    return true;
}

TorchWrap::TorchWrap(const std::string& configFileName)
    :
    m_aligner{configFileName}
{
    tinyxml2::XMLDocument doc;
    doc.LoadFile(configFileName.c_str());
    m_modelPath = doc.FirstChildElement("ModelPath")->GetText();
    m_imgDim = std::stoi(doc.FirstChildElement("ImgDimension")->GetText());
    m_repFileName = doc.FirstChildElement("RepresentationFileName")->GetText();

    m_executer = std::make_unique<Executer>(m_modelPath, m_imgDim);
}

void TorchWrap::ForwardImage(const std::string& imgPath, std::vector<float>& representation)
{
    cv::Mat src = cv::imread(imgPath, 1);
    OpenFaceCpp::DlibImage img;
    dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(src));
    dlib::rectangle face = m_aligner.GetLargestFaceBoundingBox(img);
    cv::Mat alignedFace = m_aligner.AlignImg(96, img, face);

    std::string inputImgName = "temp_aligned.jpg";
    cv::imwrite(inputImgName, alignedFace);

    m_executer->SendMsg(inputImgName);
    
    std::string imgRepStr;
    m_executer->ReceiveMsg(imgRepStr);
    
    int curPos = 0;
    auto pos = imgRepStr.find(",", curPos);
    while(pos!=std::string::npos)
    {
        std::string substr = imgRepStr.substr(curPos, pos - curPos);
        std::cout<<substr<<std::endl;
        representation.emplace_back(stof(substr));
        curPos = pos+1; 
        pos = imgRepStr.find(",", curPos);
    }
}
