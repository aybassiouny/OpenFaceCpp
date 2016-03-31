#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "NativeDLib.h"
#include "TorchWrap.h"
#include <eigen3/Eigen/Eigen>
#include <chrono>
//#include <eigenlibsvm/svm_utils.h>
//#include <eigenlibsvm/eigen_extensions.h>

using namespace std;
using namespace cv;

NativeDLib align;

//using namespace esvm;
using namespace Eigen;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
using TimePoint = std::chrono::steady_clock::time_point;

using namespace std::chrono;

int getduration(TimePoint t1, TimePoint t2)
{
	return  std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t2).count();
}

// bp::child initChild2()
// {
// 	ctx.stdout_behavior = bp::capture_stream(); 
//     ctx.stdin_behavior = bp::capture_stream();
//     ctx.environment = bp::self::get_environment();
// 	std::string exec = "/home/aybassiouny/torch/install/bin/th openface_server.lua -model models/openface/nn4.v1.t7 -imgDim 96";    
// 	std::cout<<"Launching "<<exec<<std::endl;
// 	return bp::launch_shell(exec, ctx);
// }

vector<double>  getImgRep(string imgName, TorchWrap tw, bp::child& c){
	Image img;
	TimePoint t1 = steady_clock::now();
	dlib::load_image(img, imgName);
	cout<<"load_image: "<<getduration(steady_clock::now(), t1)<<endl; t1 = steady_clock::now();
    BoundingBox face = align.getLargestFaceBoundingBox(img);
    cout<<"getLargestFaceBoundingBox: "<<getduration(steady_clock::now(), t1)<<endl; t1 = steady_clock::now();
    Mat alignedFace = align.alignImg("affine", 96, img, face, "blabla");
    cout<<"alignImg: "<<getduration(steady_clock::now(), t1)<<endl; t1 = steady_clock::now();
    cv::imwrite("temp_aligned.jpg", alignedFace);
    cout<<"imwrite: "<<getduration(steady_clock::now(), t1)<<endl; t1 = steady_clock::now();
    auto ans = tw.forwardImage("temp_aligned.jpg", std::forward<decltype(c)>(c));
    cout<<"forwardImage: "<<getduration(steady_clock::now(), t1)<<endl; t1 = steady_clock::now();
    return ans;
}

void dox5(MatrixXf &inp){
	MatrixXf tmp = inp;
	int in_numrows = inp.rows();
	int out_numrows = 5*in_numrows;
	cout<<in_numrows<<" "<<out_numrows<<endl;
	inp.resize(out_numrows, inp.cols());
	for(int i=0; i<out_numrows; i++){
		int ori_row= i%in_numrows;
		for(int j=0; j<inp.cols(); j++)
			inp(i, j) =  tmp(ori_row, j);
	}
}

float calcDist(MatrixXf X, MatrixXf Y)
{
	MatrixXf dif = (X-Y);
	cout<<dif.squaredNorm()<<endl;
	return dif.squaredNorm();
}


namespace bp = ::boost::process; 

int main(){
	cout<<"initializing ..."<<endl;
	TorchWrap tw;
	bp::child c = tw.initChild();
	align.init("models/dlib/mean.csv", "shape_predictor_68_face_landmarks.dat");
	cout<<"Loading images ..."<<endl;
	string train1imgname = "sample.jpg";
	string train2imgname = "sample2.jpg";
	vector<string> testimgNames{"sample3.png", "sample5.jpg", "sample4.jpg"};
	vector<double> train1rep, train2rep;
    train1rep = getImgRep(train1imgname, tw, c);
    train2rep = getImgRep(train2imgname, tw, c);
    vector<vector<double>> testReps;
    for(auto& imgName: testimgNames)
    	testReps.emplace_back(getImgRep(imgName, tw, c));
    cout<<"Predicting ..."<<endl;
    constexpr int numDims = 127;
    //SVMClassifier svm;

    MatrixXf gt(2,1); MatrixXf feat(2, numDims), train1Feat(1,numDims), train2Feat(1,numDims), testFeat(1,numDims);
    //MatrixXf gt(1,1); MatrixXf feat(1, numDims), testFeat(1,numDims);
    for(int i=0; i<numDims; i++){
		feat(0,i) = train1rep[i];
		train1Feat(0,i) = train1rep[i];
    }
	for(int i=0; i<numDims; i++){
		feat(1,i) = train2rep[i];
		train2Feat(0,i) = train2rep[i];
	}
	
	gt<<1,2;
	//gt<<1;
    vector<int> ans;
    dox5(feat);dox5(gt);
    //svm.train(feat, gt);
    int i=0;
    for(auto& imgRep: testReps){
    	cout<<"Doing "<<testimgNames[i++]<<endl;
    	for(int i=0; i<numDims; i++)
			testFeat(0,i) = imgRep[i];
		auto res = calcDist(testFeat, train1Feat);
		cout<<res<<endl;
		res = calcDist(testFeat, train2Feat);
		cout<<res<<endl;
  //   	cout<<testFeat<<endl;
	 //  	//svm.test(testFeat, ans);
	 //  	int Nte= testFeat.rows();
	 //    int D= testFeat.cols();
	 //    ans.resize(Nte);
	 //    struct svm_node *x = Malloc(struct svm_node, D);
	 //    for(int j=0;j<D;j++) {
	 //        x[j].index = j+1;
	 //        x[j].value = testFeat(0, j);
	 //      }
	 //    double prob_estimates[2]; // assumes 2 classes
	 //    double predict_label;
		
		// cout<<"Probability = 1"<<endl;
		// predict_label = svm_predict_probability(svm.model_, x, prob_estimates);
		// for(int i=0; i<2; i++) cout<<prob_estimates[i]<<" "; cout<<endl;
	
		// predict_label= svm_predict_values(svm.model_, x, prob_estimates);
		// cout<<"Probability = 0"<<endl;
		// for(int i=0; i<2; i++) cout<<prob_estimates[i]<<" "; cout<<endl;
		
	 //  	cout<<predict_label<<endl;
    }
}
