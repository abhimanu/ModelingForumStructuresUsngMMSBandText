
//============================================================================
// Name        : ThreadStructuredMMSBpoissonforForums.cpp
// Author      : Abhimanu Kumar
// Version     :
// Copyright   : Your copyright notice
// Description : MMSBpoisson in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdlib.h>
//#include <boost/numeric/ublas/vector.hpp>
//#include <boost/numeric/ublas/io.hpp>
//#include <boost/numeric/ublas/matrix.hpp>
//#include "csv_parser.hpp"
//#include <boost/multi_array.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include "Utils.h"
#include <math.h>

using namespace std;
//using namespace boost::numeric::ublas;


//template <class T>
//void printMat(matrix<T> *mat, int M, int N) {
//	for (int k = 0; k < M; ++k) {
//		for (int j = 0; j < N; ++j) {
//			cout << (*mat)(k,j) << "," ;
//		}
//		cout << endl;
//	}
//}
//
//
//void printMat3D(boost::multi_array<float,3> *mat, int M, int N, int P) {
//	for (int k = 0; k < M; ++k) {
//		for (int j = 0; j < N; ++j) {
//			for (int i = 0; i < P; ++i) {
//				cout << (*mat)[k][j][i] << " " ;
//			}
//			cout << "||==||" ;
//		}
//		cout << endl;
//	}
//}

class MMSBpoisson{
private:
//	boost::numeric::ublas::vector<float>* alpha;
//	boost::numeric::ublas::matrix<float>* gamma;
//	boost::multi_array<float,4>* phiPQ;
////	matrix<float>* B;
//	matrix<float>* nu;
//	matrix<float>* lambda;
//	matrix<float>* kappa;
//	matrix<float>* theta;
	int num_users;
	int K;
	Utils* utils;
//	matrix<float>* bDenomSum;
//	matrix<int>* inputMat;
//	float multiplier;
//	static const  int variationalStepCount=10;
//	static constexpr float threshold=1e-5;
//	static constexpr float alphaStepSize=1e-6;
//	static constexpr float stepSizeMultiplier=0.5;
//	static constexpr float globalThreshold=1e-4;
//
//	void initialize(int num_users, int K, matrix<int>* inputMat){
//		gamma = new matrix<float>(num_users,K);
////		B = new matrix<float>(K,K);
//		nu = new matrix<float>(K,K);
//		lambda = new matrix<float>(K,K);
//		kappa = new matrix<float>(K,K);
//		theta = new matrix<float>(K,K);
//		alpha = new boost::numeric::ublas::vector<float>(K);
//		phiPQ = new boost::multi_array<float, 4>(boost::extents[K][K][num_users][num_users]);
//
//		multiplier = alphaStepSize;
//		this->inputMat = inputMat;
//		this->num_users = num_users;
//		this->K=K;
//		initializeAlpha();
////		initializeB();
//		
//		initializeNu();
//		initializeLambda();
//		initializeKappa();
//		initializeTheta();
//		initializeGamma();
//
//		for(int k=0;k<K;k++)cout<<(*alpha)(k)<<" ";
//		cout<<endl;
////		printMat(B,K,K);
//		printMat(inputMat,num_users,num_users);
//		printMat(gamma,num_users,K);
//	};

//	void normalizePhi(){
//		cout<<"phi\n";
//		for (int i = 0; i < num_users; ++i) {
//			for (int j = 0; j < num_users; ++j) {
//				normalizePhiK(i,j);
//			}
//			cout<<endl;
//		}
//	};



public:
	MMSBpoisson(Utils *);
//	void getParameters(boost::numeric::ublas::matrix<int>* matFile, int num_users, int K, int iter_threshold, int inner_iter);
	float getUniformRandom();
//	matrix<float>* updatePhiVariational(int p, int q, float sumGamma_p, float sumGamma_q);
	float getVariationalLogLikelihood();
//	void updateB(int p, int q, matrix<float>* oldPhi_pq);
	void updateB();
	void updateGamma(int p);
	void updateNu();
	void updateNuFixedPoint(float stepSize, int iter);
	void updateLambda();
	float dataFunction(int p, int q, int g, int h);
	float updateGlobalParams(int inner_iter);
	void variationalUpdatesPhi(int p, int q);
//	float getMatrixRowSum(boost::numeric::ublas::matrix<float>* mat, int row_id, int num_cols);
	void printPhi(int p, int q);
	void printPhiFull();
	float getLnGamma(float value);

	float getDigamaValue(float value);
//	void normalizePhiK(int p, int q, bool debugPrint=false);
//	void copyAlpha(boost::numeric::ublas::vector<float>* oldAlpha);
//	void updateAlpha(bool flagLL);
	void initializeGamma();
	void initializeAlpha();
	void initializeB();
	
	void initializeNu();
	void initializeLambda();
	void initializeTheta();
	void initializeKappa();
	
//	matrix<float>* getPis();
//	boost::numeric::ublas::vector<float>* getVecH();
//	boost::numeric::ublas::vector<float>* getVecG();
};

int main(int argc, char** argv) {
//	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!



//	Utils *utilsClass = new Utils();

//	matrix<int>* matFile =  utilsClass->readCsvToMat((char *) ("18_simulatedMat.csv"), 18, 18);

	//	cout<<matFile(0,0)<<matFile(0,2)<<endl;
	
//	std::vector<T>* userList;
//	std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>, class_hash<pair<int,int>>>* userAdjlist;
//	std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>, class_hash<pair<int,int>>>* userThreadPost;

//	MMSBpoisson* mmsb = new MMSBpoisson(utilsClass);
//	mmsb->getParameters(matFile, atoi(argv[3]), atoi(argv[4]), atoi(argv[1]), atoi(argv[2]));



//	cout<<mmsb->getLnGamma(atoi(argv[1]));

//	cout<<mmsb->getDigamaValue(20)<<endl;
//	cout<<mmsb->getDigamaValue(30)<<endl;
//	cout<<mmsb->getDigamaValue(10)<<endl;
//	cout<<mmsb->getDigamaValue(1)<<endl;
//	cout<<mmsb->getDigamaValue(0)<<endl;
//	cout<<mmsb->getDigamaValue(101)<<endl;
	return 0;



}
