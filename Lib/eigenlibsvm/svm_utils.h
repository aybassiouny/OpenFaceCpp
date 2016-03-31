/*
Author: Andrej Karpathy (http://cs.stanford.edu/~karpathy/)
1 May 2012
BSD licence
*/

#ifndef __EIGEN_SVM_UTILS_H__
#define __EIGEN_SVM_UTILS_H__

#include <string>
#include <vector>
#include <libsvm/svm.h>
#include <eigen3/Eigen/Eigen>

using namespace std;
namespace esvm {
  
  /* 
  Trains a binary SVM on dense data with linear kernel using libsvm. 
  Usage:
  
  vector<int> yhat;
  SVMClassifier svm;
  svm.train(X, y);
  svm.test(X, yhat);

  where X is an Eigen::MatrixXf that is NxD array. (N D-dimensional data),
  y is a vector<int> or an Eigen::MatrixXf Nx1 vector. The labels are assumed
  to be -1 and 1. This version doesn't play nice if your dataset is 
  too unbalanced.
  */
  class SVMClassifier{
    public:
      
      SVMClassifier();
      ~SVMClassifier();
      
      // train the svm
      void train(const Eigen::MatrixXf &X, const vector<int> &y);
      void train(const Eigen::MatrixXf &X, const Eigen::MatrixXf &y);
      
      // test on new data 
      void test(const Eigen::MatrixXf &X, vector<int> &yhat);
      
      // libsvm does not directly calculate the w and b, but a set of support
      // vectors. This function will use them to compute w and b, as currenly
      // we assume linear kernel only
      // yhat = sign( X * w + b )
      void getw(Eigen::MatrixXf &w, float &b);
      
      // I/O
      int saveModel(const char *filename);
      void loadModel(const char *filename); 
      
      // options
      void setC(double Cnew); //default is 1.0
      
      //TODO: add cross validation support
      //TODO: add probability support?
      
    
      svm_model *model_;
      svm_problem *problem_;
      svm_parameter *param_;
      svm_node *x_space_;
      
      int D_; //dimension of data
  };
};


#endif //__EIGEN_SVM_UTILS_H__
