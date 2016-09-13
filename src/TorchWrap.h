#pragma once
#include <boost/process.hpp>
#include <boost/process/config.hpp>

#include <vector>
#include <cstdio>
#include <memory> 

namespace OpenFaceCpp
{
    class TorchWrap {
    public:
        TorchWrap(const std::string& configFileName);
        std::vector<double> ForwardImage(const std::string& imgPath);
        std::vector<double> ForwardImage(const std::string& imgPath, const boost::process::windows::child& child);
        boost::process::child initChild();

    private:
        std::string m_modelPath;
        int m_imgDim;
        std::shared_ptr<FILE> m_pipe;
        std::string m_inModelPath;
        std::string m_repFileName;
    };
}