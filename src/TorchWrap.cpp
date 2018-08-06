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
#if defined (BOOST_WINDOWS_API)

template<typename Child>
inline bool is_running(const Child &p, int & exit_code)
{
	DWORD code;
	//single value, not needed in the winapi.
	if (!GetExitCodeProcess(p.proc_info.hProcess, &code))
		throw std::runtime_error("GetExitCodeProcess() failed");

	if (code == 259) //windows constant STILL_ACTIVE
		return true;
	else
	{
		exit_code = code;
		return false;
	}
}

#elif defined(BOOST_POSIX_API)

tempalte<typename Child>
inline bool is_running(const Child&p, int & exit_code)
{
	int status;
	auto ret = ::waitpid(p.pid, &status, WNOHANG | WUNTRACED);

	if (ret == -1)
	{
		if (errno != ECHILD) //because it no child is running, than this one isn't either, obviously.
			throw std::runtime_error("is_running error");

		return false;
	}
	else if (ret == 0)
		return true;
	else //exited
	{
		if (WIFEXITED(status))
			exit_code = status;
		return false;
	}
}

#endif

}

TorchWrap::Executer::Executer(const std::string& modelPath, int imgDim)
	:
	m_appToChild{ create_pipe() },
	m_childToApp{ create_pipe() }
{
	m_childSink.open(m_appToChild.sink, boost::iostreams::close_handle);
	m_childSource.open(m_childToApp.source, boost::iostreams::close_handle);

	const std::string luaPath = boost::process::search_path("luajit");
	const std::string c_exec = luaPath + " openface_server.lua -model " + modelPath + " -imgDim " + std::to_string(imgDim);
	std::vector<std::string > commandWithArguments;
	StringToVecOfStrings(c_exec, commandWithArguments);

	std::cout << "Launching " << c_exec << std::endl;
	m_child = std::make_unique<boost::process::child>(
		execute(
			initializers::set_args(commandWithArguments),
			initializers::on_CreateProcess_setup([this](executor &e)
	{
		//e.startup_info.dwFlags = STARTF_RUNFULLSCREEN; 
		initializers::bind_stdout(m_childSink).on_CreateProcess_setup(e);
		initializers::bind_stdin(m_childSource).on_CreateProcess_setup(e);
		initializers::show_window(SW_NORMAL).on_CreateProcess_setup(e);
	}),
			initializers::on_CreateProcess_error([](executor&)
	{
		std::cout << GetLastError() << std::endl; })
		));

	m_appSource.open(m_appToChild.source, boost::iostreams::close_handle);
	m_childSink.open(m_childToApp.sink, boost::iostreams::close_handle);

	int errorCode;
	is_running(*m_child, errorCode);
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
	std::string result = "";
	while (result.find("time elapsed: 0ms") == std::string::npos)
	{
		std::getline(childOutAppin, result);
		if (result.find(",", 0) != std::string::npos)
		{
			msg = result;
			//break;
		}
	}
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
