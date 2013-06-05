//============================================================================
// Name        : ThreadStructuredMMSBforForums.cpp
// Author      : Abhimanu Kumar
// Version     :
// Copyright   : Your copyright notice
// Description : MMSB in C++, Ansi-style
//============================================================================

#include <iostream>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <csv_parser/csv_parser.hpp>
#include <boost/multi_array.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include "Utils.h"
#include <math.h>

using namespace std;
using namespace boost::numeric::ublas;


template <class T>
void printMat(matrix<T> *mat, int M, int N) {
	for (int k = 0; k < M; ++k) {
		for (int j = 0; j < N; ++j) {
			cout << (*mat)(k,j) << "," ;
		}
		cout << endl;
	}
}


void printMat3D(boost::multi_array<float,3> *mat, int M, int N, int P) {
	for (int k = 0; k < M; ++k) {
		for (int j = 0; j < N; ++j) {
			for (int i = 0; i < P; ++i) {
				cout << (*mat)[k][j][i] << " " ;
			}
			cout << "||==||" ;
		}
		cout << endl;
	}
}

class MMSB{
private:
	boost::numeric::ublas::vector<float>* alpha;
	matrix<float>* gamma;
	boost::multi_array<float,3>* phiPToQ;
	boost::multi_array<float,3>* phiPFromQ;
	matrix<float>* B;
	int num_users;
	int K;
	Utils* utils;
	matrix<float>* bDenomSum;
	matrix<int>* inputMat;
	float multiplier;
	static const  int variationalStepCount=10;
	static const float threshold=1e-5;
	static const float alphaStepSize=1e-6;
	static const float stepSizeMultiplier=0.5;
	static const float globalThreshold=1e-4;

	void initialize(int num_users, int K, matrix<int>* inputMat){
		gamma = new matrix<float>(num_users,K);
		B = new matrix<float>(K,K);
		alpha = new boost::numeric::ublas::vector<float>(K);
		phiPToQ = new boost::multi_array<float, 3>(boost::extents[num_users][num_users][K]);
		phiPFromQ = new boost::multi_array<float, 3>(boost::extents[num_users][num_users][K]);

		multiplier = alphaStepSize;
		this->inputMat = inputMat;
		this->num_users = num_users;
		this->K=K;
//		for (int j = 0; j < K; ++j) {
//			alpha->operator ()(j)= getUniformRandom();		// initialize alpha
//			for (int i = 0; i < num_users; ++i) {
////				gamma->operator ()(i,j)=2.0*num_users/K;	// initialize gamma
//				for (int k = 0; k < num_users; ++k) {
//					(*phiPToQ)[i][k][j] = getUniformRandom();	// initialize phiPToQ
//					(*phiPFromQ)[i][k][j] = getUniformRandom();	// initialize phiPFromQ
////					cout<< (*phi)[i][k][j] <<" ";
//				}
////				cout<<endl;
//			}
////			cout<<endl;
////			cout<<"B\n";
//			for (int i = 0; i < K; ++i) {
////				B->operator ()(i,j)= getUniformRandom();	// initialize B
////				cout<<B->operator ()(i,j)<<" ";
//			}
////			cout<<endl;
//		}
//		normalizePhi();
		initializeB();
		initializeGamma();
		initializeAlpha();
	};

	void normalizePhi(){
//		cout<<"phi\n";
		for (int i = 0; i < num_users; ++i) {
			for (int j = 0; j < num_users; ++j) {
				normalizePhiK(i,j);
			}
//			cout<<endl;
		}
	};



public:
	MMSB(Utils *);
	void getParameters(matrix<int>* matFile, int num_users, int K);
	float getUniformRandom();
	matrix<float>* updatePhiVariational(int p, int q, float sumGamma_p, float sumGamma_q);
	float getVariationalLogLikelihood();
	void updateB(int p, int q, matrix<float>* oldPhi_pq);
	void updateB();
	void updateGamma(int p, matrix<float>* oldPhi_pq, boost::numeric::ublas::vector<float>* alphaOld);
	float dataFunction(int p, int q, int g, int h);
	float getDigamaValue(float value);
	void normalizePhiK(int p, int q, bool debugPrint=false);
	void copyAlpha(boost::numeric::ublas::vector<float>* oldAlpha);
	void updateAlpha(bool flagLL);
	void initializeGamma();
	void initializeAlpha();
	void initializeB();
	matrix<float>* getPis();
	boost::numeric::ublas::vector<float>* getVecH();
	boost::numeric::ublas::vector<float>* getVecG();
};

MMSB::MMSB(Utils* utils){
	this->utils = utils;
}

void MMSB::initializeAlpha(){
	for (int k = 0; k < K; ++k) {
		alpha->operator ()(k)= getUniformRandom();
	}
}

void MMSB::normalizePhiK(int p, int q, bool debugPrint){
	float topic_sum1 = 0;
	float topic_sum2 = 0;
	for (int k = 0; k < K; ++k) {
		topic_sum1 += (*phiPToQ)[p][q][k];
		topic_sum2 += (*phiPFromQ)[p][q][k];
	}
	if(debugPrint){
		cout<<"topic_sum1 "<<topic_sum1<<endl;
		cout<<"topic_sum2 "<<topic_sum2<<endl;
	}
	for (int k = 0; k < K; ++k) {
		(*phiPToQ)[p][q][k] = ((*phiPToQ)[p][q][k])/topic_sum1;
		(*phiPFromQ)[p][q][k] = ((*phiPFromQ)[p][q][k])/topic_sum2;
		if(debugPrint){
			cout<<(*phiPToQ)[p][q][k]<<" ";
			cout<<(*phiPFromQ)[p][q][k]<<" ";
		}
	}
	if(debugPrint)
		cout<<endl;
}

float MMSB::getVariationalLogLikelihood(){
float ll=0;
	for (int p = 0; p < num_users; ++p) {
		for (int q = 0; q < num_users; ++q) {
			for (int g = 0; g < K; ++g) {
				float gammaSum =0;
				for (int h = 0; h < K; ++h) {
					ll+=((*phiPToQ)[p][q][g])*((*phiPFromQ)[p][q][h])*dataFunction(p,q,g,h);	// line 1 of MMSB
					gammaSum += gamma->operator ()(p,h);
				}
				float digammaTerm = (getDigamaValue(gamma->operator ()(p,g))-getDigamaValue(gammaSum));
				ll+=((*phiPToQ)[p][q][g])*digammaTerm;											// line 2
				ll+=((*phiPFromQ)[p][q][g])*digammaTerm;										// line 3
				ll-=(((*phiPToQ)[p][q][g])*log(((*phiPToQ)[p][q][g])));							// line 6
				ll-=(((*phiPFromQ)[p][q][g])*log(((*phiPFromQ)[p][q][g])));						// line 6
			}
		}
		float alphaSum = 0;
		float gammaSum = 0;
		for (int k = 0; k < K; ++k){
			alphaSum+=alpha->operator ()(k);
			gammaSum += gamma->operator ()(p,k);
		}
		ll+=lgamma(alphaSum);																	//line 4
		ll-=lgamma(gammaSum);																	//line 5
		for (int k = 0; k < K; ++k) {
			ll-=lgamma(alpha->operator ()(k));													//line 4
			float digammaTerm = (getDigamaValue(gamma->operator ()(p,k))-getDigamaValue(gammaSum));
			ll+= ((alpha->operator ()(k)-1)*(digammaTerm));										//line 4

			ll+=lgamma(gamma->operator ()(p,k));												//line 5
			ll-=((gamma->operator ()(p,k)-1)*(digammaTerm));									//line 5
		}
	}
	return ll;
}

float MMSB::getDigamaValue(float value){
	return boost::math::digamma(value);
}

void MMSB::getParameters(matrix<int>* inputMat, int num_users, int K){
	initialize(num_users, K, inputMat);
	getVariationalLogLikelihood();
	boost::numeric::ublas::vector<float>* oldAlpha = new boost::numeric::ublas::vector<float>(K);
	copyAlpha(oldAlpha);
	float newLL = 0;//getVariationalLogLikelihood();
	float oldLL = 0;
	int iter=0;
	do{
		cout<<"iter "<<iter++<<endl;
		oldLL = newLL;
		for (int p = 0; p < num_users; ++p) {
			float sumGamma_p =0;
			for (int k = 0; k < K; ++k) {
				sumGamma_p += gamma->operator ()(p,k);
			}
			matrix<float>* oldPhi_pq;
			for (int q = 0; q < num_users; ++q) {
				float sumGamma_q = 0;
				for (int k = 0; k < K; ++k) {
					sumGamma_q += gamma->operator ()(q,k);
				}

				oldPhi_pq = updatePhiVariational(p,q,sumGamma_p,sumGamma_q);

				// partial update gamma and B
//				updateGamma(p, oldPhi_pq, oldAlpha);
//				updateB(p,q,oldPhi_pq);
			}
			updateGamma(p, oldPhi_pq, oldAlpha);
//			updateB();
		}
		// update alpha
//		updateGamma(p, oldPhi_pq, oldAlpha);
		updateB();
		newLL= getVariationalLogLikelihood();
		updateAlpha(oldLL>newLL);
//		cout<<abs(oldLL-newLL)<<endl;
//		cout.precision(6);
		cout<<setprecision(9)<<newLL<<endl;
	}while(abs(oldLL-newLL)>globalThreshold);
	matrix<float>* pi = getPis();
	cout<<"PI\n";
	printMat(pi,num_users,K);
	cout<<"B\n";
	printMat(B,K,K);
}

matrix<float>* MMSB::getPis(){
	matrix<float>* pi = new matrix<float>(num_users,K);
	for (int p = 0; p < num_users; ++p) {
		float sumPk = 0;
		for (int k = 0; k < K; ++k) {
			sumPk+=((*gamma)(p,k));
		}
		for (int k = 0; k < K; ++k) {
			((*pi)(p,k)) = ((*gamma)(p,k))/sumPk;
		}
	}
	return pi;
}

void MMSB::copyAlpha(boost::numeric::ublas::vector<float>* copyAlpha){
	for (int k = 0; k < K; ++k) {
		copyAlpha->operator ()(k) = alpha->operator ()(k);
	}
}

void MMSB::updateAlpha(bool flagLL){
	// this part is coded by approximating H as a diagnoal + rank one matrix (Blei's LDA paper)
	boost::numeric::ublas::vector<float>* hVec = getVecH();
	float sumAlpha = 0;
	for (int k = 0; k < K; ++k) {
		sumAlpha+=alpha->operator ()(k);
	}
	float z = -num_users*utils->trigamma(sumAlpha);
	float gByhSum = 0, sumHinv=0;
	boost::numeric::ublas::vector<float>* gVec = getVecG();
	for (int k = 0; k < K; ++k) {
		gByhSum += ((gVec->operator ()(k))/(hVec->operator ()(k)));
		sumHinv +=(1.0/(hVec->operator ()(k)));
	}
	float c = gByhSum/((1.0/z)+sumHinv);

	if(!flagLL)
		multiplier=multiplier*stepSizeMultiplier;
	for (int k = 0; k < K; ++k) {
		alpha->operator ()(k) = alpha->operator ()(k) - ((gVec->operator ()(k) - c)/hVec->operator ()(k))*multiplier;
	}
}

boost::numeric::ublas::vector<float>* MMSB::getVecG(){					// gradient of L w.r.t alpha
	boost::numeric::ublas::vector<float>* vecG = new boost::numeric::ublas::vector<float>(K);
	boost::numeric::ublas::vector<float> vecSumGammaP(num_users);
	float sumAlpha = 0;
	for (int k = 0; k < K; ++k) {
		sumAlpha+=alpha->operator ()(k);
	}
	for (int p = 0; p < num_users; ++p) {
		vecSumGammaP(p)=0;
		for (int k = 0; k < K; ++k) {
			vecSumGammaP(p)+=gamma->operator ()(p,k);
		}
	}
	float sumPgradient=0;
	for (int k = 0; k < K; ++k) {
		vecG->operator ()(k) = num_users*(getDigamaValue(sumAlpha)-getDigamaValue(alpha->operator ()(k)));
		sumPgradient = 0;
		for (int p = 0; p < num_users; ++p) {
			sumPgradient += (getDigamaValue(gamma->operator ()(p,k)) - getDigamaValue(vecSumGammaP(p)));
		}
		vecG->operator ()(k) = vecG->operator ()(k) + sumPgradient;
	}
	return vecG;
}

boost::numeric::ublas::vector<float>* MMSB::getVecH(){					// hessian of L w.r.t alpha
	boost::numeric::ublas::vector<float>* vecH = new boost::numeric::ublas::vector<float>(K);
	for (int k = 0; k < K; ++k) {
		vecH->operator ()(k)= num_users*(utils->trigamma(alpha->operator ()(k)));
	}
	return vecH;
}


void MMSB::initializeB(){
	bDenomSum = new matrix<float>(K,K);
//	for (int g = 0; g < K; ++g) {
//		for (int h = 0; h < K; ++h) {
//			float temp_gh=0;
//			float sum_pq_gh =0;
//			for (int p = 0; p < num_users; ++p) {
//				for (int q = 0; q < num_users; ++q) {
//					temp_gh+=((*inputMat)(p,q))*((*phiPToQ)[p][q][g])*((*phiPFromQ)[p][q][h]);
//					sum_pq_gh+=((*phiPToQ)[p][q][g])*((*phiPFromQ)[p][q][h]);
//				}
//			}
//			B->operator ()(g,h)=temp_gh/sum_pq_gh;
//			bDenomSum->operator ()(g,h) = sum_pq_gh;
//		}
//	}
	for (int g = 0; g < K; ++g) {
		for (int h = 0; h < K; ++h) {
			(*B)(g,h)=getUniformRandom()*0.01;
		}
		(*B)(g,g)=(1-(*B)(g,g));
	}
	printMat(B,K,K);
	printMat(inputMat,num_users,K);
}


void MMSB::updateB(){
	for (int g = 0; g < K; ++g) {
		for (int h = 0; h < K; ++h) {
			float temp_gh=0;
			float sum_pq_gh =0;
			for (int p = 0; p < num_users; ++p) {
				for (int q = 0; q < num_users; ++q) {
					temp_gh+=(*inputMat)(p,q)*((*phiPToQ)[p][q][g])*((*phiPFromQ)[p][q][h]);
					sum_pq_gh+=((*phiPToQ)[p][q][g])*((*phiPFromQ)[p][q][h]);
				}
			}
			B->operator ()(g,h)=temp_gh/sum_pq_gh;
		}
	}
}

void MMSB::updateB(int p, int q, matrix<float>* oldPhi_pq){
	//optimize this shit later
	float oldValSum=0, newValSum=0, newDenominator=0;

	for (int g = 0; g < K; ++g) {
		for (int h = 0; h < K; ++h) {
//			oldValSum=0; newValSum=0; newDenominator=0;
//			oldValSum=(B->operator ()(g,h))*(bDenomSum->operator ()(g,h));
//			newValSum=oldValSum-((*inputMat)(p,q)*(oldPhi_pq->operator ()(0,g))*(oldPhi_pq->operator ()(0,h)));
//			newValSum= newValSum + ((*inputMat)(p,q)*((*phiPToQ)[p][q][g])*((*phiPFromQ)[p][q][h]));
//			newDenominator = bDenomSum->operator ()(g,h)-((oldPhi_pq->operator ()(0,g))*(oldPhi_pq->operator ()(0,h)));
//			newDenominator = newDenominator+(((*phiPToQ)[p][q][g])*((*phiPFromQ)[p][q][h]));
//			bDenomSum->operator ()(g,h) = newDenominator;
////			cout<<"newDenominator "<<newDenominator<<endl;
//			if(isnan(newDenominator)){
//				cout<<oldValSum<<" "<<newValSum<<" "<<newDenominator<<" "<<endl;
//				cout<<(*inputMat)(p,q)<<" "<<(*phiPToQ)[p][q][g]<<" "<<(*phiPFromQ)[p][q][h]<<endl;
//				cout<<oldPhi_pq->operator ()(0,g)<<" "<<oldPhi_pq->operator ()(0,h)<<" ";
//				cout<<bDenomSum->operator ()(g,h);
//				cout<<endl;
////				printB(B,K);
//				printMat(inputMat,num_users,num_users);
//				printMat(bDenomSum,K,K);
//			}
//			float newVal = (newValSum/newDenominator);
//			if(isnan(newVal)){
//				cout<<"Got you Bitch"<<endl;
//			}
//			B->operator ()(g,h) = newVal;

			updateB();

//			float temp_gh=0;
//			float sum_pq_gh =0;
//			for (int p = 0; p < num_users; ++p) {
//				for (int q = 0; q < num_users; ++q) {
//					temp_gh+=(*inputMat)(p,q)*((*phiPToQ)[p][q][g])*((*phiPFromQ)[p][q][h]);
//					sum_pq_gh+=((*phiPToQ)[p][q][g])*((*phiPFromQ)[p][q][h]);
//				}
//			}
//			B->operator ()(g,h)=temp_gh/sum_pq_gh;
		}
	}
}



void MMSB::initializeGamma(){
	float sumPtoQ=0;
	float sumPfromQ=0;

	for (int p = 0; p < num_users; ++p) {
		for (int k = 0; k < K; ++k) {
//			sumPtoQ=0;
//			sumPfromQ=0;
//			for (int q = 0; q < num_users; ++q) {
//				sumPtoQ+= (*phiPToQ)[p][q][k];
//				sumPfromQ+=(*phiPFromQ)[p][q][k];
//			}
//			gamma->operator ()(p,k)=alpha->operator ()(k)+sumPtoQ+sumPfromQ;
			gamma->operator ()(p,k)=2.0*num_users/K;
		}
	}
}


void MMSB::updateGamma(int p, matrix<float>* oldPhi_pq, boost::numeric::ublas::vector<float>* oldAlpha){
	float sumPtoQ=0;
	float sumPfromQ=0;
	//optimize this shit later
	for (int k = 0; k < K; ++k) {
		sumPtoQ=0;
		sumPfromQ=0;
		for (int q = 0; q < num_users; ++q) {
			sumPtoQ+= (*phiPToQ)[p][q][k];
			sumPfromQ+=(*phiPFromQ)[p][q][k];
		}
		gamma->operator ()(p,k)=alpha->operator ()(k)+sumPtoQ+sumPfromQ;
	}
}


matrix<float>* MMSB::updatePhiVariational(int p, int q, float sumGamma_p, float sumGamma_q){
	matrix<float>* oldPhi_pq = new matrix<float>(2,K);
	for (int k = 0; k < K; ++k){ 						// initialize phi_p->q and phi_p<-q
		oldPhi_pq->operator ()(0,k)=(*phiPToQ)[p][q][k];
		oldPhi_pq->operator ()(1,k)=(*phiPFromQ)[p][q][k];
//		cout<<oldPhi_pq->operator ()(0,k)<<" "<<oldPhi_pq->operator ()(0,k)<<" ";
		(*phiPToQ)[p][q][k] = 1.0/K;
		(*phiPFromQ)[p][q][k] = 1.0/K;
	}
//	cout<<endl;
	boost::numeric::ublas::vector<float> tempOldPhiPToQ(K);
	boost::numeric::ublas::vector<float> tempOldPhiPFromQ(K);

//	float newLL = getVariationalLogLikelihood();
//	float oldLL = 0;
	int counter =0;
	do{
//		oldLL = newLL;
		for (int k = 0; k < K; ++k) { // update old values
			tempOldPhiPToQ(k) = ((*phiPToQ)[p][q][k]);
			tempOldPhiPFromQ(k) = ((*phiPFromQ)[p][q][k]);
		}

		for (int g = 0; g < K; ++g) {					// p->q update
			float digammaTerm = (getDigamaValue(gamma->operator ()(p,g))-getDigamaValue(sumGamma_p));
			float logDataExpectation = 0;
			for (int k = 0; k < K; ++k) {
				logDataExpectation+= (tempOldPhiPFromQ(k)*dataFunction(p,q,g,k));
						//(((*phiPFromQ)[p][q][k])*dataFunction(p,q,g,k));
//				cout<<"hello "<<digammaTerm<<" "<<logDataExpectation<<" "<<dataFunction(p,q,g,k)<<" "<<tempOldPhiPFromQ(k)<<" "<<p<<q<<k<<endl;
			}

			(*phiPToQ)[p][q][g] =exp(digammaTerm+logDataExpectation);
		}
		for (int h = 0; h < K; ++h) {
			float digammaTerm = (getDigamaValue(gamma->operator ()(q,h))-getDigamaValue(sumGamma_q));
			float logDataExpectation =0;
			for (int k = 0; k < K; ++k) {
				logDataExpectation += (tempOldPhiPToQ(k)*dataFunction(p,q,k,h));
						//(((*phiPToQ)[p][q][k])*dataFunction(p,q,h,k));
			}
			(*phiPFromQ)[p][q][h] =exp(digammaTerm+logDataExpectation);
//			if(isnan((*phiPFromQ)[p][q][h])){
//				cout<<
//			}
		}
		counter++;
		normalizePhiK(p,q);
//		newLL= getVariationalLogLikelihood();
//		if(counter>variationalStepCount){
//			break;
//			cout<<"Variational "<<abs(oldLL-newLL)<<" p q "<<p<<" "<<q<<endl;
//		}
	}while(counter>variationalStepCount);//(abs(oldLL-newLL)>threshold);

	return oldPhi_pq;
}

float MMSB::dataFunction(int p, int q, int g, int h){
	float retVal1 = ((*inputMat)(p,q)*log(B->operator ()(g,h)));
	float retVal2 = ((1-(*inputMat)(p,q))*log(1-B->operator ()(g,h)));
//	cout<<"B(g,h)"<<B->operator ()(g,h)<<endl;

	float returnVal = retVal1 + retVal2;
//	cout<<"2nd"<<returnVal<<" "<<retVal1<<" "<<retVal2<<endl;
	return returnVal;
}


float MMSB::getUniformRandom(){
	boost::mt19937 rng;
	rng.seed(static_cast<unsigned int>(std::time(0)));
//	rng.distribution().reset();
	static boost::uniform_01<boost::mt19937> zeroone(rng);
	return zeroone();

}




int main() {
//	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!



	Utils *utilsClass = new Utils();

	matrix<int>* matFile =  utilsClass->readCsvToMat((char *) ("monkLK.csv"), 18, 18);
//	cout<<matFile(0,0)<<matFile(0,2)<<endl;

	MMSB* mmsb = new MMSB(utilsClass);
	mmsb->getParameters(matFile, 18, 4);
//	cout<<mmsb->getDigamaValue(20)<<endl;
//	cout<<mmsb->getDigamaValue(30)<<endl;
//	cout<<mmsb->getDigamaValue(10)<<endl;
//	cout<<mmsb->getDigamaValue(1)<<endl;
//	cout<<mmsb->getDigamaValue(0)<<endl;
//	cout<<mmsb->getDigamaValue(101)<<endl;
	return 0;



}
