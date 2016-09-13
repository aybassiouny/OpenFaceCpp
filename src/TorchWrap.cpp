#include "TorchWrap.h"

#include <iostream>
#include <string>
#include <chrono>
#include "tinyxml2.h"

using namespace OpenFaceCpp;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
using TimePoint = std::chrono::steady_clock::time_point;

using namespace std::chrono;
using namespace boost::process;

TorchWrap::TorchWrap(const std::string& configFileName)
{
    tinyxml2::XMLDocument doc;
    doc.LoadFile(configFileName.c_str());
    m_modelPath = doc.FirstChildElement("ModelPath")->GetText();
    m_imgDim = std::stoi(doc.FirstChildElement("ImgDimension")->GetText());
    m_repFileName = doc.FirstChildElement("RepresentationFileName")->GetText();
}

child TorchWrap::initChild()
{
    /*ctx.stdout_behavior = bp::capture_stream(); 
    ctx.stdin_behavior = bp::capture_stream();
    ctx.environment = bp::self::get_environment();*/
    std::string exec = "/home/aybassiouny/torch/install/bin/th openface_server.lua -model "+m_modelPath+" -imgDim "+std::to_string(m_imgDim);    
    //std::string exec = "/home/aybassiouny/torch/install/bin/th openface_server.lua -model models/openface/nn4.v1.t7 -imgDim 96";    
    std::cout<<"Launching "<<exec<<std::endl;
    return execute(windows::initializers::run_exe(exec));
    //return bp::launch_shell(exec, ctx);
}

std::vector<double> TorchWrap::ForwardImage(const std::string& imgPath)
{
    std::string exec = "th openface_server.lua -model "+m_modelPath+" -imgDim "+std::to_string(m_imgDim);    
    std::cout<<"Launching "<<exec<<std::endl;
    
    std::vector<double> imgRep;
    std::cout<<"listening ..."<<std::endl;
    std::ofstream out(m_repFileName);
    
    std::string imgRepStr = "";
    int curPos = 0;
    auto pos = imgRepStr.find(",", curPos);
    while(pos!=std::string::npos){
        std::string substr = imgRepStr.substr(curPos, pos - curPos);
        out<<substr<<std::endl;
        imgRep.emplace_back(stod(substr));
        curPos = pos+1; 
        pos = imgRepStr.find(",", curPos);
    }
    return imgRep;
}
    
std::vector<double> TorchWrap::ForwardImage(const std::string& imgPath, const boost::process::windows::child& ch)
{
    std::vector<double> imgRep;
    std::cout<<"listening ..."<<std::endl;
    //bp::pistream &is = ch.get_stdout(); 
    //bp::postream &pout = ch.get_stdin();
    std::ofstream out(m_repFileName);
    
    std::string imgRepStr = "";
    std::cout<<"sending ..."<<imgPath<<std::endl;
    //pout<<imgPath<<std::endl;
    std::cout<<"receiving ..."<<std::endl;
    //std::getline(is, imgRepStr);
    int curPos = 0;
    auto pos = imgRepStr.find(",", curPos);
    while(pos!=std::string::npos){
        std::string substr = imgRepStr.substr(curPos, pos - curPos);
        out<<substr<<std::endl;
        imgRep.emplace_back(stod(substr));
        curPos = pos+1; 
        pos = imgRepStr.find(",", curPos);
    }

    return imgRep;
}
