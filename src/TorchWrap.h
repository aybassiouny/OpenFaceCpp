#ifndef TORCHWRAP_H
#define TORCHWRAP_H


#include <vector>
#include <string>
#include <iostream>
#include <cstdio>
#include <memory> 
#include <string>
#include <exception> 
#include <fstream>
#include <boost/process.hpp> 
#include <vector> 

namespace bp = ::boost::process; 

class TorchWrap{
	std:: string modelPath; 
    int imgDim;    
	std::shared_ptr<FILE> pipe;
	std::string inModelPath;
	std::string repFileName;
public:
	TorchWrap();
	TorchWrap(std::string inModelPath, int inImgDim, std::string inRepFileName);	
	std::vector<double> 	forwardImage(std::string imgPath, bp::child& c);
	bp::child initChild();
	//boost::optional<bp::child> c;

	bp::context ctx; 
	//void forwardPath(std::string imgPath);
};

#endif