#include "TorchWrap.h"
#include <chrono>

using namespace std;
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
using TimePoint = std::chrono::steady_clock::time_point;

using namespace std::chrono;

int getduration3(TimePoint t1, TimePoint t2)
{
    return  std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t2).count();
}


TorchWrap::TorchWrap() :TorchWrap("models/openface/nn4.v1.t7", 96, "representation.txt")
{
	
}

TorchWrap::TorchWrap(std::string inModelPath, int inImgDim, std::string inRepFileName)
{
	modelPath = inModelPath; 
    imgDim  = inImgDim;    
    repFileName = inRepFileName;
}

bp::child TorchWrap::initChild()
{
	ctx.stdout_behavior = bp::capture_stream(); 
    ctx.stdin_behavior = bp::capture_stream();
    ctx.environment = bp::self::get_environment();
	std::string exec = "/home/aybassiouny/torch/install/bin/th openface_server.lua -model "+modelPath+" -imgDim "+std::to_string(imgDim);    
	//std::string exec = "/home/aybassiouny/torch/install/bin/th openface_server.lua -model models/openface/nn4.v1.t7 -imgDim 96";    
	std::cout<<"Launching "<<exec<<std::endl;
	return bp::launch_shell(exec, ctx);
}
	
std::vector<double> TorchWrap::forwardImage(std::string imgPath, bp::child& c)
{
	//std::string exec = "/home/aybassiouny/torch/install/bin/th openface_server.lua -model "+modelPath+" -imgDim "+std::to_string(imgDim);    
	//std::cout<<"Launching "<<exec<<std::endl;
	//bp::child c =  bp::launch_shell(exec, ctx);
	std::vector<double> imgRep;
	std::cout<<"listening ..."<<std::endl;
	bp::pistream &is = c.get_stdout(); 
    bp::postream &pout = c.get_stdin();
    TimePoint t1 = steady_clock::now();
    std::ofstream out(repFileName);
    
	//std::string cmd = "th openface_server.lua -model "+modelPath+" -imgDim "+std::to_string(imgDim)+" -imgPath "+imgPath;
    //cout<<"before cmd: "<<getduration3(steady_clock::now(), t1)<<endl; t1 = steady_clock::now();
    //pipe = std::shared_ptr<FILE>(popen(cmd.c_str(), "r"), pclose);
    //std::cout<<cmd<<std::endl;
    //cout<<"after cmd: "<<getduration3(steady_clock::now(), t1)<<endl; t1 = steady_clock::now();
    //if (!pipe) 
	//	throw std::exception();
	//char buffer[128];
    std::string imgRepStr = "";
    //std::cout<<"sending ..."<<imgPath<<std::endl;
    pout<<imgPath<<endl;
    //std::cout<<"receiving ..."<<std::endl;
	std::getline(is, imgRepStr);
	//std::cout<<"Received imgRepStr of length "<<imgRepStr.size()<<std::endl;
    // cout<<"before pipe: "<<getduration3(steady_clock::now(), t1)<<endl; t1 = steady_clock::now();
    // while (!feof(pipe.get())) {
    //     if (fgets(buffer, 128, pipe.get()) != NULL)
    //         imgRepStr += buffer;
    // }
    cout<<"after pipe: "<<getduration3(steady_clock::now(), t1)<<endl; t1 = steady_clock::now();
    int curPos = 0;
    auto pos = imgRepStr.find(",", curPos);
    while(pos!=std::string::npos){
        std::string substr = imgRepStr.substr(curPos, pos - curPos);
        out<<substr<<std::endl;
        imgRep.emplace_back(stod(substr));
        curPos = pos+1; 
        pos = imgRepStr.find(",", curPos);
    }
    cout<<"str manip: "<<getduration3(steady_clock::now(), t1)<<endl; t1 = steady_clock::now();
    return imgRep;
}