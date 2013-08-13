//============================================================================
// Name        : ThreadStructuredMMSBpoissonforForums.cpp
// Author      : Abhimanu Kumar
// Version     :
// Copyright   : Your copyright notice
// Description : MMSBpoisson in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdlib.h>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include "csv_parser.hpp"
#include <boost/multi_array.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include <unordered_map>
#include <unordered_set>

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

template <class T>
void printNanInMat(matrix<T> *mat, int M, int N) {
	cout<<"In printNanInMat\t";
	for (int k = 0; k < M; ++k) {
		for (int j = 0; j < N; ++j) {
			if(std::isnan((*mat)(k,j)))
				cout << (*mat)(k,j) << ","<<k<<","<<j<<"||\t";
		}
	}
	cout << endl;
}

void testDataStructures(std::unordered_set<int>* userList, 
		std::unordered_set<int>* threadList,
		std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* userAdjlist, 
		std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost){
	int u1;
	
    cout<< "num users "<<userList->size()<<"; "<<"num threads "<<threadList->size()<<endl;

//	cout<< "Users discovered:\t";
//	for(std::unordered_set<int>::iterator it=userList->begin(); it!=userList->end(); ++it)
//		cout<<(*it)<<"\t";
//	cout<<endl;
//
//	cout<< "Threads discovered:\t";
//	for(std::unordered_set<int>::iterator it=threadList->begin(); it!=threadList->end(); ++it)
//		cout<<(*it)<<"\t";
//	cout<<endl;
//
	
	int num_nonzeros=0;
	cout<<"User >< User >< Thread >< Count:\t";
	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<pair<int,int>>>::iterator it1=userAdjlist->begin(); it1!=userAdjlist->end(); ++it1){
		for(std::unordered_map<int,int>::iterator it2 = it1->second->begin(); it2!=it1->second->end(); ++it2){
			cout<<it1->first.first<<" >< "<<it2->first<<" >< "<<it1->first.second<<" >< "<<
				it2->second<<":\t";
			num_nonzeros++;
		}
	}
	cout<<"\tnum_nonzeros "<<num_nonzeros<<endl;
//
//	cout<<"User >< Thread >< Words:: \t";
//
//	for(std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>::iterator it1=userThreadPost->begin(); it1!=userThreadPost->end(); it1++){
//		cout<<it1->first.first<<" >< "<<it1->first.second<<" >< ";
//		for(std::vector<int>::iterator it2=it1->second->begin(); it2!=it1->second->end(); it2++)
//			cout<<" "<<*it2;
//		cout<<endl;
//	}

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

class MMSBpoisson{
private:
	boost::numeric::ublas::vector<float>* alpha;
	matrix<float>* gamma;
	unordered_map<int,int>* userIndexMap;			// <index user> pair
    std::unordered_set<int>* threadList;                                                                           
	std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* userAdjlist;
    std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost;        
//	boost::multi_array<float,4>* phiPQ;
//	matrix<float>* B;

    //Phi terms
	matrix<float>* phi_gh_sum;
	matrix<float>* phi_y_gh_sum;
	matrix<float>* phi_qh_sum;
	matrix<float>* phi_pg_sum;
	matrix<float>* phi_lgammaPhi;
	float phi_logPhi;
	

	matrix<float>* nu;
	matrix<float>* lambda;
	matrix<float>* kappa;
	matrix<float>* theta;
	int num_users;
	int K;            
	int nuIter;
	Utils* utils;
	matrix<float>* bDenomSum;
//	matrix<int>* inputMat;
	float multiplier;
	static const  int variationalStepCount=10;
	static constexpr float threshold=1e-5;
	static constexpr float alphaStepSize=1e-6;
	static constexpr float stepSizeMultiplier=0.5;
	static constexpr float globalThreshold=1e-4;

public:
	MMSBpoisson(Utils *);
	void getParameters(int iter_threshold, int inner_iter,int nu_iter);
	float getUniformRandom();
//	matrix<float>* updatePhiVariational(int p, int q, float sumGamma_p, float sumGamma_q);
	float getVariationalLogLikelihood();
//	void updateB(int p, int q, matrix<float>* oldPhi_pq);
	void updateB();
	void updateGamma(int p);
	void updateNu();
	void updateNuFixedPoint(float stepSize);
	void updateLambda();
	float dataFunction(int g, int h);
	float dataFunctionPhiUpdates(int g, int h, int Y_pq);
	float updateGlobalParams(int inner_iter);
	void variationalUpdatesPhi(int p, int q, int Y_pq, int thread_id);
	float getMatrixRowSum(matrix<float>* mat, int row_id, int num_cols);
	void printPhi(int p, int q);
	void printPhiFull();
	float getLnGamma(float value);

	float getDigamaValue(float value);
//	void normalizePhiK(int p, int q, bool debugPrint=false);
//	void copyAlpha(boost::numeric::ublas::vector<float>* oldAlpha);
//	void updateAlpha(bool flagLL);
	void initializeGamma();
	void initializeAlpha();
	void initialize(int K, std::unordered_set<int>* userList, std::unordered_set<int>* threadList,                      
		std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* userAdjlist,
		std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost);        
	void initializeB();

	void initializeAllPhiMats();
	void initializeUserIndex(std::unordered_set<int>* userList);
	
	void initializeNu();
	void initializeLambda();
	void initializeTheta();
	void initializeKappa();
	
	matrix<float>* getPis();
	boost::numeric::ublas::vector<float>* getVecH();
	boost::numeric::ublas::vector<float>* getVecG();
};

MMSBpoisson::MMSBpoisson(Utils* utils){
//	cout<<"In MMSB constructor"<<endl;
	this->utils = utils;
}

void MMSBpoisson::initializeAlpha(){
	for (int k = 0; k < K; ++k) {
		(*alpha)(k)= 0.5+(getUniformRandom()-0.5)*0.1;
	}
}

void MMSBpoisson::initialize(int K, std::unordered_set<int>* userList, std::unordered_set<int>* threadList,  
		std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* userAdjlist,
		std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost){        

	this->num_users = userList->size();//num_users;
	cout<<"num_users "<<num_users<<endl;
	this->K=K;
	gamma = new matrix<float>(num_users,K);

	userIndexMap = new unordered_map<int,int>();

	//		B = new matrix<float>(K,K);
	nu = new matrix<float>(K,K);
	lambda = new matrix<float>(K,K);
	kappa = new matrix<float>(K,K);
	theta = new matrix<float>(K,K);
	alpha = new boost::numeric::ublas::vector<float>(K);
	//		phiPQ = new boost::multi_array<float, 4>(boost::extents[K][K][num_users][num_users]);

	phi_gh_sum = new matrix<float>(K,K);
	phi_y_gh_sum = new matrix<float>(K,K);
	phi_qh_sum = new matrix<float>(num_users,K);
	phi_pg_sum = new matrix<float>(num_users,K);
	phi_lgammaPhi = new matrix<float>(K,K);
	phi_logPhi = 0;

	this->threadList = threadList;
	this->userAdjlist = userAdjlist;
	this->userThreadPost = userThreadPost;
	initializeUserIndex(userList);

//    cout<< "Hello there!"<<endl;

	multiplier = alphaStepSize;
	//		this->inputMat = inputMat;
//	this->num_users = userList->size();//num_users;
//	cout<<"num_users "<<num_users<<endl;
//	this->K=K;
	initializeAlpha();
	//		initializeB();

	initializeNu();
	initializeLambda();
	initializeKappa();
	initializeTheta();
	initializeGamma();

	cout<<"After intializeGamma\n";
	for(int k=0;k<K;k++)cout<<(*alpha)(k)<<" ";
	cout<<endl;
	//		printMat(B,K,K);
	//		printMat(inputMat,num_users,num_users);
//	printMat(gamma,num_users,K);
	cout<<"Fag end of initialize()\n";
}

float MMSBpoisson::dataFunction(int g,int h){
//	return (*inputMat)(p,q)*log((*B)(g,h)) + (1-(*inputMat)(p,q))*log((1-(*B)(g,h)));
	return (*phi_y_gh_sum)(g,h)*(log((*lambda)(g,h)) + getDigamaValue((*nu)(g,h))) 
		-(*lambda)(g,h)*(*nu)(g,h)*(*phi_gh_sum)(g,h) - (*phi_lgammaPhi)(g,h);
}

float MMSBpoisson::dataFunctionPhiUpdates(int g , int h, int Y_pq){
	return Y_pq*(log((*lambda)(g,h)) + getDigamaValue((*nu)(g,h))) 
		-(*lambda)(g,h)*(*nu)(g,h) - getLnGamma(Y_pq+1);
}

float MMSBpoisson::getVariationalLogLikelihood(){                // TODO: change phi coz it is K*K*N*N 
	float ll=0;
	cout<<"In log-likelihood calculation "<<ll<<endl;
	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
			ll += (((*kappa)(g,h)-1)*(log((*lambda)(g,h))+getDigamaValue((*nu)(g,h))) 
					- (*nu)(g,h)*(*lambda)(g,h)/(*theta)(g,h) - (*kappa)(g,h)*log((*theta)(g,h)) 
					- lgamma((*kappa)(g,h)));
//			cout<<ll<<endl;
			ll -= (((*nu)(g,h)-1)*(log((*lambda)(g,h))+getDigamaValue((*nu)(g,h))) 
					- (*nu)(g,h) - (*nu)(g,h)*log((*lambda)(g,h)) 
					- lgamma((*kappa)(g,h)));
//			cout<<ll<<endl;

			ll += dataFunction(g,h);
//			cout<<dataFunction(g,h)<<ll<<endl;
		}
	}
	cout<<"after first for loop log-likelihood calculation "<<ll<<endl;
	for (int p = 0; p < num_users; ++p) {
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

			ll+= ((*phi_pg_sum)(p,k))*digammaTerm;                                  // line 2//newPhis
			ll+= ((*phi_qh_sum)(p,k))*digammaTerm;                                  // line 3//newPhis
		}

	}
	ll-=phi_logPhi;																				//line 8  //newPhis
//	cout<<"End of log-likelihood calculation"<<endl;
	return ll;
}

float MMSBpoisson::getDigamaValue(float value){
	return boost::math::digamma(value);
}

float MMSBpoisson::getLnGamma(float value){
	return boost::math::lgamma(value);
}

void MMSBpoisson::getParameters(int iter_threshold, int inner_iter, int nu_iter){ // TODO: change phi coz it is K*K*N*N 
//	initialize(num_users, K);//, inputMat);			// should be called from the main function
	cout<<"ll-0"<<getVariationalLogLikelihood()<<endl;
//	boost::numeric::ublas::vector<float>* oldAlpha = new boost::numeric::ublas::vector<float>(K);
//	copyAlpha(oldAlpha);
	float newLL = 0;//getVariationalLogLikelihood();
	float oldLL = 0;
	int iter=0;
	this->nuIter = nu_iter;
//	int iter_threshold = 30;
//	int inner_iter = 4;
	do{
        iter++;
		cout<<"iter "<<iter<<endl;
		oldLL=newLL;
		newLL=updateGlobalParams(inner_iter);
		if(iter>=iter_threshold)
			break;
	}while(1);//abs(oldLL-newLL)>globalThreshold);
	matrix<float>* pi = getPis();
//	cout<<"PI\n";
//	printMat(pi,num_users,K);
//	cout<<"Gamma\n";
//	printMat(gamma,num_users,K);
//	cout<<"Nu\n";
//	printMat(nu,K,K);
//	cout<<"Lambda\n";
//	printMat(lambda,K,K);

}


float MMSBpoisson::updateGlobalParams(int inner_iter){//(gamma,B,alpha,Y,inner_iter){
	float ll=0;
//	int nuIters=3;
	float nuStepSize=1e-5;
	for(int i=1; i<=inner_iter;i++){
		//		cout<<"i "<<i<<endl;
		initializeAllPhiMats();							//this is very important if we are not storing giant Phi mat
		int num_nonzeros =0;
		for(int p=0; p<num_users; p++){
			for(unordered_set<int>::iterator it=threadList->begin(); it!=threadList->end(); it++){
				for(int q=0; q<num_users; q++){
					if(p==q)
						continue;
					int Y_pq=0;
					pair<int,int> user_thread = std::make_pair(userIndexMap->at(p),*it);
					if(userAdjlist->count(user_thread)>0){
						int userid_q = userIndexMap->at(q);
						if(userAdjlist->at(user_thread)->count(userid_q)>0){
							Y_pq=userAdjlist->at(user_thread)->at(userid_q);
							//cout<<"Y_pq "<<Y_pq<<"("<<user_thread.first<<","<<userid_q<<","<<user_thread.second<<")"<<"; ";
//							num_nonzeros++;
						}
					}
					variationalUpdatesPhi(p,q,Y_pq,*it);//

					//				cout<<"p,q "<<p<<" "<<q<<"||";
					//				printPhi(p,q);
				}
			}
		}   
//		cout<<"phi_lgammaPhi\n";
//		printMat(phi_lgammaPhi,K,K);
		cout<<"phi_pg_sum\n";
		printNanInMat(phi_pg_sum,num_users,K);
//		cout<<"phi_qh_sum\n";
//		printMat(phi_qh_sum,num_users,K);
		//        cout<<"update Gamma and B now"<<endl;
		//		printPhiFull();
//		cout<<"\tnum_nonzeros "<<num_nonzeros<<endl;
		for(int p=0; p<num_users; p++){
			updateGamma(p);
			//			cout<<"update B now"<<endl;
			//			updateB();
			//			cout<<"updated both B and Gamma"<<endl;
		}
		//		cout<<"Gamma\n";
		//		printMat(gamma, num_users, K);
		updateNuFixedPoint(nuStepSize);
		//		cout<<"Nu\n";
		//		printMat(nu,K,K);
		updateLambda();
		//		cout<<"Lambda\n";
		//		printMat(lambda,K,K);
		ll = getVariationalLogLikelihood();
		cout<<ll<<endl;
		//       pi = gamma./repmat(sum(gamma,2),1,K)
	}
	return ll;

}

void MMSBpoisson::initializeAllPhiMats(){
	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
			(*phi_gh_sum)(g,h) = 0;
			(*phi_y_gh_sum)(g,h)=0;
			(*phi_lgammaPhi)(g,h)=0;
		}
		for(int p=0; p<num_users; p++){
			(*phi_pg_sum)(p,g) = 0;
			(*phi_qh_sum)(p,g) = 0;
		}
	}
	phi_logPhi=0;
}


/*
 * This method needs to incorporate the thread structure t
 *
 * DO WE NOT UPDATE NU and FIX IT AT 2?
 *
 * */

void MMSBpoisson::updateNuFixedPoint(float stepSize){
	for(int it=0; it<nuIter; it++){
		for(int g=0; g<K; g++){
			for(int h=0; h<K; h++){
				float phi_nu = 0;
				float invTriGamma = 1/utils->trigamma((*nu)(g,h));
				float trigamma_gh = utils->trigamma((*nu)(g,h));
//				for(int p=0; p<K; p++){
//					for(int q=0; q<K; q++){						// TODO put t structure here
				phi_nu+=((*phi_y_gh_sum)(g,h)*trigamma_gh - (*phi_gh_sum)(g,h)*(*lambda)(g,h));//newPhis
//					}
//				}
				phi_nu+=(((*kappa)(g,h) - (*nu)(g,h))*trigamma_gh + (1 - ((*lambda)(g,h)/(*theta)(g,h)))); 
				(*nu)(g,h) = (*nu)(g,h) + stepSize*phi_nu;
			}
		}
	}
}

//void MMSBpoisson::updateNu(){
//   for(int g=0; g<K; g++){
//	  for(int h=0; h<K; h++){
//		  float phi_nu = 0;
//		  float invTriGamma = 1/utils->trigamma((*nu)(g,h));
//		  for(int p=0; p<K; p++){
//			  for(int q=0; q<K; q++){						// TODO put t structure here
//				phi_nu+=((*phiPQ)[g][h][p][q]* ((*inputMat)(g,h) - (*lambda)(g,h)*invTriGamma));//newPhis(same as above)
//			  }
//		  }
//		  phi_nu+=((*kappa)(g,h) + (1 - ((*lambda)(g,h)/(*theta)(g,h)))*invTriGamma); 
//		  (*nu)(g,h) = phi_nu;
//	  }
//   }
//}

/*
 * This method needs to incorporate the thread structure t
 * */

void MMSBpoisson::updateLambda(){
	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
//			float phi_gh=0;
//			float phi_y_gh=0;
//			for(int p=0; p<num_users; p++){
//				for(int q=0; q<num_users; q++){	//TODO: put a loop for thread t here as well
//				   phi_gh+=((*phiPQ)[g][h][p][q]);
//				   phi_y_gh+=(((*phiPQ)[g][h][p][q])*(*inputMat)(p,q));
//				}
//			}
			(*lambda)(g,h) = ((*phi_y_gh_sum)(g,h)+(*kappa)(g,h))/(((*phi_gh_sum)(g,h) + 1/(*theta)(g,h))*(*nu)(g,h));  //newPhis
		}
	}
}

/*
 * This method needs to incorporate the thread structure t
 * */
/*
 * the poisson counts are mostly b/w 0-5
 * we need rate parameter to be in (2,3)
 * for that gamma's (shape)nu=2 (scale)lambda=2.0
 * */


void MMSBpoisson::initializeNu(){
	for(int g=0; g<K; g++)
		for(int h=0; h<K; h++)
			(*nu)(g,h)=2;
}

void MMSBpoisson::initializeLambda(){
	for(int g=0; g<K; g++)
		for(int h=0; h<K; h++)
			(*lambda)(g,h)=3;
}

void MMSBpoisson::initializeTheta(){
	for(int g=0; g<K; g++)
		for(int h=0; h<K; h++)
			(*theta)(g,h)=3;
}

void MMSBpoisson::initializeKappa(){
	for(int g=0; g<K; g++)
		for(int h=0; h<K; h++)
			(*kappa)(g,h)=2;
}




//void MMSBpoisson::printPhi(int p, int q){
//	for(int g=0;g<K; g++){
//		for(int h=0;h<K;h++)
//			cout<<(*phiPQ)[g][h][p][q]<<" ";
//		cout<<";";
//	}
//	cout<<"||||||||||||||";
//}


//void MMSBpoisson::printPhiFull(){
//	cout<<endl<<endl;
//	for(int p=0; p<num_users; p++){
//		for(int q=0; q<num_users; q++){
//			cout<<"p,q "<<p<<" "<<q<<"||";
//			for(int g=0;g<K; g++){
//				for(int h=0;h<K;h++)
//					cout<<(*phiPQ)[g][h][p][q]<<" ";
//				cout<<";";
//			}
//			cout<<"||||||||||||||";
//		}
//	}
//	cout<<endl;
//}


void MMSBpoisson::variationalUpdatesPhi(int p, int q, int Y_pq, int thread_id){
	//N = size(Y,1);
	//K = size(alpha,2);
//	boost::numeric::ublas::vector<float>* deriv_phi_p = new boost::numeric::ublas::vector<float>(K);
//	boost::numeric::ublas::vector<float>* deriv_phi_q = new boost::numeric::ublas::vector<float>(K);
//	int Y_pq = (*inputMat)(p,q);
	float digamma_p_sum = getDigamaValue(getMatrixRowSum(gamma,p,K));
	float digamma_q_sum = getDigamaValue(getMatrixRowSum(gamma,q,K));
	

	matrix<float>* phi_gh_pq = new matrix<float>(K,K);
	boost::numeric::ublas::vector<float>* phi_pg_update = new boost::numeric::ublas::vector<float>(K);
	boost::numeric::ublas::vector<float>* phi_qh_update = new boost::numeric::ublas::vector<float>(K);
//	cout<<digamma_q_sum<<endl;
//	cout<<digamma_p_sum<<endl;

	float phi_sum = 0;
	for(int g=0;g<K;g++){
		for(int h=0;h<K;h++){
			(*phi_gh_pq)(g,h) = exp(dataFunctionPhiUpdates(g,h,Y_pq) 
				+ (getDigamaValue((*gamma)(p,g)) - digamma_p_sum)
				+ (getDigamaValue((*gamma)(q,h)) - digamma_q_sum));
			phi_sum+=(*phi_gh_pq)(g,h);

//			if(std::isnan((*phi_gh_pq)(g,h))){
//				cout<<p<<" "<<q<<" "<<Y_pq<<" "<<thread_id<<" "<<(*phi_gh_pq)(g,h)<<endl;
//			}
		}
		(*phi_qh_update)(g) = 0;
		(*phi_pg_update)(g) = 0;
	}

//	cout<<"variationalUpdatesPhi"<<endl;
	
	float temp_phi_gh = 0;

	for(int g=0;g<K;g++){
		for(int h=0;h<K;h++){
			temp_phi_gh=(*phi_gh_pq)(g,h);
			(*phi_gh_pq)(g,h) = ((*phi_gh_pq)(g,h))/phi_sum ;
			(*phi_gh_sum)(g,h) += ((*phi_gh_pq)(g,h));
			(*phi_y_gh_sum)(g,h) += ((*phi_gh_pq)(g,h)*Y_pq);//((*inputMat)(p,q)));
			(*phi_lgammaPhi)(g,h) += ((*phi_gh_pq)(g,h)*lgamma(Y_pq+1));//(*inputMat)(p,q)+1));
			phi_logPhi += ((*phi_gh_pq)(g,h)*log((*phi_gh_pq)(g,h)));

			if(std::isnan((*phi_gh_pq)(g,h)))
				cout<<p<<" "<<q<<" "<<Y_pq<<" "<<thread_id<<" "<<(*phi_gh_pq)(g,h)<<" "<<temp_phi_gh<<" "<<dataFunctionPhiUpdates(g,h,Y_pq)<<" "<< exp(dataFunctionPhiUpdates(g,h,Y_pq))<<endl;
//			if(h==0){
//				(*phi_pg_update)(g)=(*phi_gh_pq)(g,h);
//			}else{
//				(*phi_pg_update)(g)+=((*phi_gh_pq)(g,h));
//			}
//
//			if(g==0){
//				(*phi_qh_update)(h)=(*phi_gh_pq)(g,h);
//			}else{
//				(*phi_qh_update)(h)+=((*phi_gh_pq)(g,h));
//			}

//			cout<<(*phiPQ)[g][h][p][q]<<" ";
		}
	}

	for(int h=0; h<K; h++){
		for(int g=0; g<K; g++){
			(*phi_qh_update)(h) += (*phi_gh_pq)(g,h);
		}
	}

	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
			(*phi_pg_update)(g) += (*phi_gh_pq)(g,h);
		}
	}

	for(int k=0; k<K; k++){
		(*phi_pg_sum)(p,k) += ((*phi_pg_update)(k));
		(*phi_qh_sum)(q,k) += ((*phi_qh_update)(k));
	}

	delete phi_gh_pq;
	delete phi_pg_update;
	delete phi_qh_update;
//	cout<<endl;
}


float MMSBpoisson::getMatrixRowSum(matrix<float>* mat, int row_id, int num_cols){
	float row_sum = 0;
	for(int j=0; j<num_cols; j++)
		row_sum+=(*mat)(row_id,j);
	return row_sum;
}

matrix<float>* MMSBpoisson::getPis(){
	matrix<float>* pi = new matrix<float>(num_users,K);
	for (int p = 0; p < num_users; ++p) {
		float sumPk = 0;		for (int k = 0; k < K; ++k) {
			sumPk+=((*gamma)(p,k));
		}
		for (int k = 0; k < K; ++k) {
			((*pi)(p,k)) = ((*gamma)(p,k))/sumPk;
		}
	}
	return pi;
}


//boost::numeric::ublas::vector<float>* MMSBpoisson::getVecG(){					// gradient of L w.r.t alpha
//	boost::numeric::ublas::vector<float>* vecG = new boost::numeric::ublas::vector<float>(K);
//	boost::numeric::ublas::vector<float> vecSumGammaP(num_users);
//	float sumAlpha = 0;
//	for (int k = 0; k < K; ++k) {
//		sumAlpha+=alpha->operator ()(k);
//	}
//	for (int p = 0; p < num_users; ++p) {
//		vecSumGammaP(p)=0;
//		for (int k = 0; k < K; ++k) {
//			vecSumGammaP(p)+=gamma->operator ()(p,k);
//		}
//	}
//	float sumPgradient=0;
//	for (int k = 0; k < K; ++k) {
//		vecG->operator ()(k) = num_users*(getDigamaValue(sumAlpha)-getDigamaValue(alpha->operator ()(k)));
//		sumPgradient = 0;
//		for (int p = 0; p < num_users; ++p) {
//			sumPgradient += (getDigamaValue(gamma->operator ()(p,k)) - getDigamaValue(vecSumGammaP(p)));
//		}
//		vecG->operator ()(k) = vecG->operator ()(k) + sumPgradient;
//	}
//	return vecG;
//}
//
//boost::numeric::ublas::vector<float>* MMSBpoisson::getVecH(){					// hessian of L w.r.t alpha
//	boost::numeric::ublas::vector<float>* vecH = new boost::numeric::ublas::vector<float>(K);
//	for (int k = 0; k < K; ++k) {
//		vecH->operator ()(k)= num_users*(utils->trigamma(alpha->operator ()(k)));
//	}
//	return vecH;
//}


void MMSBpoisson::initializeUserIndex(unordered_set<int>* userList){  
	int index=0;
	for(std::unordered_set<int>::iterator it=userList->begin(); it!=userList->end(); it++){
		userIndexMap->insert({index, (*it)});
		index++;
	}	
}

void MMSBpoisson::initializeGamma(){
	cout<<"In InitializeGamma\n";
	for (int p = 0; p < num_users; ++p) {
//		cout<<p<<" ";
		for (int k = 0; k < K; ++k) {
			(*gamma)(p,k)=(*alpha)(k)+(getUniformRandom()-0.5)*0.1;
//			cout<<(*alpha)(k)<<" "<<(*gamma)(p,k)<<" ";

		}
//		cout<<endl;
	}
}


void MMSBpoisson::updateGamma(int p){
	float gamma_pk=0;
	for(int k=0; k<K; k++){
		gamma_pk = (*alpha)(k);
//		for(int q=0; q<num_users; q++){
////			cout<<"k,q "<<k<<" "<<q<<"|";
//			if(p==q)
//				continue;
//			for(int h=0; h<K; h++){
//				gamma_pk+= ((*phiPQ)[k][h][p][q] + (*phiPQ)[h][k][q][p]);
//			}
//		}
		gamma_pk+= ((*phi_pg_sum)(p,k) + (*phi_qh_sum)(p,k));	// new Phis
		(*gamma)(p,k)=gamma_pk;
//		cout<<"gamma_pk "<<p<<","<<k<<","<<gamma_pk<<";";
	}
//	cout<<endl;
}




float MMSBpoisson::getUniformRandom(){
	boost::mt19937 rng;
	rng.seed(static_cast<unsigned int>(std::time(0)));
//	rng.distribution().reset();
	static boost::uniform_01<boost::mt19937> zeroone(rng);
	return zeroone();

}




int main(int argc, char** argv) {
//	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!


	int K=atoi(argv[2]);
	Utils *utilsClass = new Utils();
//	int i = 2349834748;

	matrix<int>* matFile =  utilsClass->readCsvToMat((char *) ("18_simulatedMat.csv"), 18, 18);

	cout<<(*matFile)(0,0)<<(*matFile)(0,2)<<endl;
	

	std::unordered_set<int>* userList = new std::unordered_set<int>();
	std::unordered_set<int>* threadList = new std::unordered_set<int>();
	std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<pair<int,int>>>* userAdjlist = 
		new std::unordered_map<std::pair<int,int>,std::unordered_map<int,int>*, class_hash<pair<int,int>>>();
	std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost = 
		new std::unordered_map<std::pair<int,int>,std::vector<int>*, class_hash<pair<int,int>>>();

//	username_mention_graph.txt

	utilsClass->readThreadStructureFile(argv[1], userList, threadList, userAdjlist, userThreadPost);

//	testDataStructures(userList,threadList, userAdjlist,userThreadPost);

//	cout<<endl<<i<<" "<<INT_MAX<<endl;

//	cout<<"Before MMSB constructor"<<endl;
	MMSBpoisson* mmsb = new MMSBpoisson(utilsClass);

//	cout<<"after MMSB constructor call"<<endl;
//	mmsb->getParameters(matFile, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));

	mmsb->initialize(K, userList, threadList, userAdjlist, userThreadPost);	
	mmsb->getParameters(atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
	delete userList;
	delete userAdjlist;
	delete threadList;
	delete userThreadPost;



//	cout<<mmsb->getLnGamma(atoi(argv[1]));

//	cout<<mmsb->getDigamaValue(20)<<endl;
//	cout<<mmsb->getDigamaValue(30)<<endl;
//	cout<<mmsb->getDigamaValue(10)<<endl;
//	cout<<mmsb->getDigamaValue(1)<<endl;
//	cout<<mmsb->getDigamaValue(0)<<endl;
//	cout<<mmsb->getDigamaValue(101)<<endl;
	return 0;



}
