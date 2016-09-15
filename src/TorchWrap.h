#pragma once
#include <boost/process.hpp>
#include <boost/process/config.hpp>

#include <vector>
#include <cstdio>
#include <memory> 
#include "NativeDLib.h"

namespace OpenFaceCpp
{
    class TorchWrap {
    public:
        TorchWrap(const std::string& configFileName);
        void ForwardImage(const std::string& imgPath, std::vector<float>& representation);

    private:
        class Executer
        {
        public:
            Executer(const std::string& modelPath, int imgDim);

            bool SendMsg(const std::string& msg);
            bool ReceiveMsg(std::string& msg);

        private:
            std::unique_ptr<boost::process::child> m_child;

            boost::process::pipe m_appToChild;
            boost::process::pipe m_childToApp;

            boost::iostreams::file_descriptor_sink m_childSink;
            boost::iostreams::file_descriptor_sink m_appSink;
            boost::iostreams::file_descriptor_source m_childSource;
            boost::iostreams::file_descriptor_source m_appSource;
        };

        OpenFaceCpp::NativeDLib m_aligner;
        std::unique_ptr<Executer> m_executer;

        std::string m_modelPath;
        std::shared_ptr<FILE> m_pipe;
        std::string m_inModelPath;
        std::string m_repFileName;

        int m_imgDim;
    };
}
