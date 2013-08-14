//============================================================================
// Name        : ThreadStructuredMMSBpoissonforForums.cpp
// Author      : Abhimanu Kumar
// Version     :
// Copyright   : Your copyright notice
// Description : MMSBpoisson in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdlib.h>
#include <cstdlib>
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

template <class T>
void printNegInMat(matrix<T> *mat, int M, int N) {
	cout<<"In printNegInMat\t";
	for (int k = 0; k < M; ++k) {
		for (int j = 0; j < N; ++j) {
			if(((*mat)(k,j))<=0)
				cout << (*mat)(k,j) << ","<<k<<","<<j<<"||\t";
		}
	}
	cout << endl;
}

void testDataStructures(std::unordered_map<int,int>* userList, 
		std::unordered_set<int>* threadList,
		std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* userAdjlist, 
		std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost){
	int u1;
	
    cout<< "num users "<<userList->size()<<"; "<<"num threads "<<threadList->size()<<endl;

	cout<< "Users discovered:\t";
	for(std::unordered_map<int,int>::iterator it=userList->begin(); it!=userList->end(); ++it)
		cout<<(it->first)<<" : "<<it->second<<"\t";
	cout<<endl;

	cout<< "Threads discovered:\t";
	for(std::unordered_set<int>::iterator it=threadList->begin(); it!=threadList->end(); ++it)
		cout<<(*it)<<"\t";
	cout<<endl;

	
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

void printMat3D(boost::multi_array<double,3> *mat, int M, int N, int P) {
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
	boost::numeric::ublas::vector<double>* alpha;
	matrix<double>* gamma;
	unordered_map<int,int>* userIndexMap;			// <index user> pair
    std::unordered_set<int>* threadList;                                                                           
    std::unordered_map<int,int>* userList;                                                                           
	std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* userAdjlist;
    std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost;        
	
	// heldout set
	std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* heldUserAdjlist;
//	boost::multi_array<double,4>* phiPQ;
//	matrix<double>* B;

	matrix<double>* held_phi_gh_sum;
	matrix<double>* held_phi_y_gh_sum;
	matrix<double>* held_phi_lgammaPhi;
    
	//Phi terms
	matrix<double>* phi_gh_sum;
	matrix<double>* phi_y_gh_sum;
	matrix<double>* phi_qh_sum;
	matrix<double>* phi_pg_sum;
	matrix<double>* phi_lgammaPhi;
	double phi_logPhi;
	

	matrix<double>* nu;
	matrix<double>* lambda;
	matrix<double>* kappa;
	matrix<double>* theta;
	int num_users;
	int K;            
	int nuIter;
	double stepSizeNu=0;
	Utils* utils;
	matrix<double>* bDenomSum;
//	matrix<int>* inputMat;
	double multiplier;
	static const  int variationalStepCount=10;
	static constexpr double threshold=1e-5;
	static constexpr double alphaStepSize=1e-6;
	static constexpr double stepSizeMultiplier=0.5;
	static constexpr double globalThreshold=1e-4;

public:
	MMSBpoisson(Utils *);
	void getParameters(int iter_threshold, int inner_iter,int nu_iter);
	double getUniformRandom();
//	matrix<double>* updatePhiVariational(int p, int q, double sumGamma_p, double sumGamma_q);
	double getVariationalLogLikelihood();
//	void updateB(int p, int q, matrix<double>* oldPhi_pq);
	void updateB();
	void updateGamma(int p);
	void updateNu();
	void updateNuFixedPoint();
	void updateLambda();
	double dataFunction(int g, int h);
	double dataFunctionPhiUpdates(int g, int h, int Y_pq);
	double updateGlobalParams(int inner_iter);
	void variationalUpdatesPhi(int p, int q, int Y_pq, int thread_id);
	double getMatrixRowSum(matrix<double>* mat, int row_id, int num_cols);
	void printPhi(int p, int q);
	void printPhiFull();
	double getLnGamma(double value);

	double getDigamaValue(double value);
//	void normalizePhiK(int p, int q, bool debugPrint=false);
//	void copyAlpha(boost::numeric::ublas::vector<double>* oldAlpha);
//	void updateAlpha(bool flagLL);
	void initializeGamma();
	void initializeAlpha();
	void initialize(int K, std::unordered_map<int,int>* userList, std::unordered_set<int>* threadList,                      
		std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* userAdjlist,
		std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* heldUserAdjlist,
		std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost,
		double stepSizeNu);        
	void initializeB();

	void initializeAllPhiMats();
	void initializeUserIndex(std::unordered_map<int,int>* userList);
	
	void initializeNu();
	void initializeLambda();
	void initializeTheta();
	void initializeKappa();
	
	double getHeldoutLogLikelihood();
	
	matrix<double>* getPis();
	boost::numeric::ublas::vector<double>* getVecH();
	boost::numeric::ublas::vector<double>* getVecG();
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

void MMSBpoisson::initialize(int K, std::unordered_map<int,int>* userList, std::unordered_set<int>* threadList,  
		std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* userAdjlist,
		std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* heldUserAdjlist,
		std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost,
		double stepSizeNu){        

    this->userList = userList;
	this->num_users = userList->size();	// this stays the same even with heldout as we donot delete from original
	cout<<"num_users stepSizeNu "<<num_users<<" "<<stepSizeNu<<endl;
	this->K=K;            
	this->stepSizeNu=stepSizeNu;

	gamma = new matrix<double>(num_users,K);

	userIndexMap = new unordered_map<int,int>();

	//		B = new matrix<double>(K,K);
	nu = new matrix<double>(K,K);
	lambda = new matrix<double>(K,K);
	kappa = new matrix<double>(K,K);
	theta = new matrix<double>(K,K);
	alpha = new boost::numeric::ublas::vector<double>(K);
	//		phiPQ = new boost::multi_array<double, 4>(boost::extents[K][K][num_users][num_users]);
	held_phi_gh_sum = new matrix<double>(K,K);
	held_phi_y_gh_sum = new matrix<double>(K,K);
	held_phi_lgammaPhi = new matrix<double>(K,K);

	phi_gh_sum = new matrix<double>(K,K);
	phi_y_gh_sum = new matrix<double>(K,K);
	phi_qh_sum = new matrix<double>(num_users,K);
	phi_pg_sum = new matrix<double>(num_users,K);
	phi_lgammaPhi = new matrix<double>(K,K);
	phi_logPhi = 0;

	this->threadList = threadList;
	this->userAdjlist = userAdjlist;
	this->heldUserAdjlist = heldUserAdjlist;
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

//	cout<<"After intializeGamma\n";
	for(int k=0;k<K;k++)cout<<(*alpha)(k)<<" ";
	cout<<endl;
//	printMat(gamma,num_users,K);
//	cout<<"Fag end of initialize()\n";
}

double MMSBpoisson::getHeldoutLogLikelihood(){
	double ll=0;
	double phi_sum = 0;
	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
			(*held_phi_y_gh_sum)(g,h)=0;
			(*held_phi_gh_sum)(g,h)=0;
			(*held_phi_lgammaPhi)(g,h) =0;
		}
	}
	
	matrix<double>* phi_gh_pq = new matrix<double>(K,K);
	
	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<pair<int,int>>>::iterator it1=heldUserAdjlist->begin(); it1!=heldUserAdjlist->end(); ++it1){
		for(std::unordered_map<int,int>::iterator it2 = it1->second->begin(); it2!=it1->second->end(); ++it2){
//			cout<< "In heldoutLog-Likeli\n";
			int p = userList->at(it1->first.first);
			int q = userList->at(it2->first);
//			cout<< "In heldoutLog-Likelii after p,q index access\n";
			double digamma_p_sum = getDigamaValue(getMatrixRowSum(gamma,p,K));
			double digamma_q_sum = getDigamaValue(getMatrixRowSum(gamma,q,K));
			int Y_pq = it2->second;
			double ph_sum=0;
			for(int g=0;g<K;g++){
				for(int h=0;h<K;h++){
					(*phi_gh_pq)(g,h) = exp(dataFunctionPhiUpdates(g,h,Y_pq) 
							+ (getDigamaValue((*gamma)(p,g)) - digamma_p_sum)
							+ (getDigamaValue((*gamma)(q,h)) - digamma_q_sum));
					phi_sum += (*phi_gh_pq)(g,h);

//					ll+=(Y_pq*log((*lambda)(g,h)) - (*lambda)(g,h)-lgamma(Y_pq+1));
				}
			}
			for(int g=0;g<K;g++){
				for(int h=0;h<K;h++){
					(*phi_gh_pq)(g,h) = (*phi_gh_pq)(g,h)/phi_sum;          
					(*held_phi_gh_sum)(g,h)+=(*phi_gh_pq)(g,h);
					(*held_phi_y_gh_sum)(g,h)+=((*phi_gh_pq)(g,h)*Y_pq);
					(*held_phi_lgammaPhi)(g,h) += ((*phi_gh_pq)(g,h)*lgamma(Y_pq+1));//(*inputMat)(p,q)+1));
				}
			}
		}
	}


	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
			ll+=((*held_phi_y_gh_sum)(g,h)*(log((*lambda)(g,h)) + getDigamaValue((*nu)(g,h))) 
				-(*lambda)(g,h)*(*nu)(g,h)*(*held_phi_gh_sum)(g,h) - (*held_phi_lgammaPhi)(g,h));
		}
	}

    delete phi_gh_pq;

	return ll;
}



double MMSBpoisson::dataFunction(int g,int h){
//	return (*inputMat)(p,q)*log((*B)(g,h)) + (1-(*inputMat)(p,q))*log((1-(*B)(g,h)));
	return (*phi_y_gh_sum)(g,h)*(log((*lambda)(g,h)) + getDigamaValue((*nu)(g,h))) 
		-(*lambda)(g,h)*(*nu)(g,h)*(*phi_gh_sum)(g,h) - (*phi_lgammaPhi)(g,h);
}

double MMSBpoisson::dataFunctionPhiUpdates(int g , int h, int Y_pq){
	return Y_pq*(log((*lambda)(g,h)) + getDigamaValue((*nu)(g,h))) 
		-(*lambda)(g,h)*(*nu)(g,h) - getLnGamma(Y_pq+1);
}

double MMSBpoisson::getVariationalLogLikelihood(){                // TODO: change phi coz it is K*K*N*N 
	double ll=0;
//	cout<<"In log-likelihood calculation "<<ll<<endl;
	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
//			cout<<"log((*lambda)(g,h))"<<log((*lambda)(g,h))<<" ((*lambda)(g,h))"<<((*lambda)(g,h));
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
//	cout<<"after first for loop log-likelihood calculation "<<ll<<endl;
	for (int p = 0; p < num_users; ++p) {
		double alphaSum = 0;
		double gammaSum = 0;
		for (int k = 0; k < K; ++k){
			alphaSum+=alpha->operator ()(k);
			gammaSum += gamma->operator ()(p,k);
		}
		ll+=lgamma(alphaSum);																	//line 4
		ll-=lgamma(gammaSum);																	//line 5
		for (int k = 0; k < K; ++k) {
			ll-=lgamma(alpha->operator ()(k));													//line 4
			double digammaTerm = (getDigamaValue(gamma->operator ()(p,k))-getDigamaValue(gammaSum));
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

double MMSBpoisson::getDigamaValue(double value){
	return boost::math::digamma(value);
}

double MMSBpoisson::getLnGamma(double value){
	return boost::math::lgamma(value);
}

void MMSBpoisson::getParameters(int iter_threshold, int inner_iter, int nu_iter){ // TODO: change phi coz it is K*K*N*N 
//	initialize(num_users, K);//, inputMat);			// should be called from the main function
	cout<<"ll-0"<<getVariationalLogLikelihood()<<endl;
//	boost::numeric::ublas::vector<double>* oldAlpha = new boost::numeric::ublas::vector<double>(K);
//	copyAlpha(oldAlpha);
	double newLL = 0;//getVariationalLogLikelihood();
	double oldLL = 0;
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
	matrix<double>* pi = getPis();
	cout<<"PI\n";
	printMat(pi,num_users,K);
	cout<<"Gamma\n";
	printMat(gamma,num_users,K);
	cout<<"Nu\n";
	printMat(nu,K,K);
	cout<<"Lambda\n";
	printMat(lambda,K,K);

}


double MMSBpoisson::updateGlobalParams(int inner_iter){//(gamma,B,alpha,Y,inner_iter){
	double ll=0;
//	int nuIters=3;
	for(int i=1; i<=inner_iter;i++){
//		cout<<"i "<<i<<endl;
		initializeAllPhiMats();							//this is very important if we are not storing giant Phi mat
		int num_nonzeros =0;
		for(int p=0; p<num_users; p++){
			for(unordered_set<int>::iterator it=threadList->begin(); it!=threadList->end(); it++){
				for(int q=0; q<num_users; q++){
//					cout<<"userid_p "<<p<<"\n";
					pair<int,int> user_thread = std::make_pair(userIndexMap->at(p),*it);
//					cout<<"userid_q "<<q<<"\n";
					int userid_q = userIndexMap->at(q);
//					cout<<"userid_q "<<userid_q<<"\n";
					//TODO also check whether it is in heldout
					if(p==q || (heldUserAdjlist->count(user_thread)>0 
								&& heldUserAdjlist->at(user_thread)->count(userid_q)==0) )
						continue;
					int Y_pq=0;
					if(userAdjlist->count(user_thread)>0){
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
//		printNanInMat(phi_lgammaPhi,K,K);
//		printNegInMat(phi_lgammaPhi,K,K);
//		cout<<"phi_pg_sum\n";
//		printNanInMat(phi_pg_sum,num_users,K);
//		printNegInMat(phi_pg_sum,num_users,K);
//		cout<<"phi_qh_sum\n";
//		printNanInMat(phi_qh_sum,num_users,K);
//		printNegInMat(phi_qh_sum,num_users,K);
//		cout<<"phi_gh_sum\n";
//		printNanInMat(phi_gh_sum,K,K);
//		printNegInMat(phi_gh_sum,K,K);
//		cout<<"phi_y_gh_sum\n";
//		printNanInMat(phi_y_gh_sum,K,K);
//		printNegInMat(phi_y_gh_sum,K,K);
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
		updateNuFixedPoint();
		updateLambda();
//		cout<<"Ello\n";
//		cout<<"gamma\n";
//		printNanInMat(gamma,num_users,K);
//		printNegInMat(gamma,num_users,K);
//		cout<<"Nu\n";
//		printNanInMat(nu,K,K);
//		printNegInMat(nu,K,K);
//		cout<<"Lambda\n";
//		printNanInMat(lambda,K,K);
//		printNegInMat(lambda,K,K);
		ll = getVariationalLogLikelihood();
        cout<<"held-ll "<<getHeldoutLogLikelihood()<<"\t ll ";
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

void MMSBpoisson::updateNuFixedPoint(){
	for(int it=0; it<nuIter; it++){
		for(int g=0; g<K; g++){
			for(int h=0; h<K; h++){
				double phi_nu = 0;
				double invTriGamma = 1/utils->trigamma((*nu)(g,h));
				double trigamma_gh = utils->trigamma((*nu)(g,h));
				phi_nu+=((*phi_y_gh_sum)(g,h)*trigamma_gh - (*phi_gh_sum)(g,h)*(*lambda)(g,h));//newPhis
//				cout<<"updateNuFixedPoint \n";
//				cout<<"nu: "<<phi_nu<<" "<<(*phi_y_gh_sum)(g,h)<<" "<<(*phi_gh_sum)(g,h)<<" ";
				phi_nu+=(((*kappa)(g,h) - (*nu)(g,h))*trigamma_gh + (1 - ((*lambda)(g,h)/(*theta)(g,h)))); 
//				cout<<(((*kappa)(g,h) - (*nu)(g,h))*trigamma_gh + (1 - ((*lambda)(g,h)/(*theta)(g,h))))<<endl; 
				(*nu)(g,h) = (*nu)(g,h) + stepSizeNu*phi_nu;
			}
		}
	}
}

//void MMSBpoisson::updateNu(){
//   for(int g=0; g<K; g++){
//	  for(int h=0; h<K; h++){
//		  double phi_nu = 0;
//		  double invTriGamma = 1/utils->trigamma((*nu)(g,h));
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
//			double phi_gh=0;
//			double phi_y_gh=0;
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
//	boost::numeric::ublas::vector<double>* deriv_phi_p = new boost::numeric::ublas::vector<double>(K);
//	boost::numeric::ublas::vector<double>* deriv_phi_q = new boost::numeric::ublas::vector<double>(K);
//	int Y_pq = (*inputMat)(p,q);
	double digamma_p_sum = getDigamaValue(getMatrixRowSum(gamma,p,K));
	double digamma_q_sum = getDigamaValue(getMatrixRowSum(gamma,q,K));
	

	matrix<double>* phi_gh_pq = new matrix<double>(K,K);
	boost::numeric::ublas::vector<double>* phi_pg_update = new boost::numeric::ublas::vector<double>(K);
	boost::numeric::ublas::vector<double>* phi_qh_update = new boost::numeric::ublas::vector<double>(K);
//	cout<<digamma_q_sum<<endl;
//	cout<<digamma_p_sum<<endl;

	double phi_sum = 0;
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
	
	double temp_phi_gh = 0;

	for(int g=0;g<K;g++){
		for(int h=0;h<K;h++){
			temp_phi_gh=(*phi_gh_pq)(g,h);
			(*phi_gh_pq)(g,h) = ((*phi_gh_pq)(g,h))/phi_sum ;
			(*phi_gh_sum)(g,h) += ((*phi_gh_pq)(g,h));
			(*phi_y_gh_sum)(g,h) += ((*phi_gh_pq)(g,h)*Y_pq);//((*inputMat)(p,q)));
			(*phi_lgammaPhi)(g,h) += ((*phi_gh_pq)(g,h)*lgamma(Y_pq+1));//(*inputMat)(p,q)+1));
			phi_logPhi += ((*phi_gh_pq)(g,h)*log((*phi_gh_pq)(g,h)));

			if(std::isnan((*phi_gh_pq)(g,h))||(*phi_gh_pq)(g,h)<=0)
				cout<<"VariationalUpdate "<<p<<" "<<q<<" "<<Y_pq<<" "<<thread_id<<" "<<(*phi_gh_pq)(g,h)<<" "<<temp_phi_gh<<" "<<dataFunctionPhiUpdates(g,h,Y_pq)<<" "<< exp(dataFunctionPhiUpdates(g,h,Y_pq))<<endl;
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


double MMSBpoisson::getMatrixRowSum(matrix<double>* mat, int row_id, int num_cols){
	double row_sum = 0;
	for(int j=0; j<num_cols; j++)
		row_sum+=(*mat)(row_id,j);
	return row_sum;
}

matrix<double>* MMSBpoisson::getPis(){
	matrix<double>* pi = new matrix<double>(num_users,K);
	for (int p = 0; p < num_users; ++p) {
		double sumPk = 0;		for (int k = 0; k < K; ++k) {
			sumPk+=((*gamma)(p,k));
		}
		for (int k = 0; k < K; ++k) {
			((*pi)(p,k)) = ((*gamma)(p,k))/sumPk;
		}
	}
	return pi;
}


//boost::numeric::ublas::vector<double>* MMSBpoisson::getVecG(){					// gradient of L w.r.t alpha
//	boost::numeric::ublas::vector<double>* vecG = new boost::numeric::ublas::vector<double>(K);
//	boost::numeric::ublas::vector<double> vecSumGammaP(num_users);
//	double sumAlpha = 0;
//	for (int k = 0; k < K; ++k) {
//		sumAlpha+=alpha->operator ()(k);
//	}
//	for (int p = 0; p < num_users; ++p) {
//		vecSumGammaP(p)=0;
//		for (int k = 0; k < K; ++k) {
//			vecSumGammaP(p)+=gamma->operator ()(p,k);
//		}
//	}
//	double sumPgradient=0;
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
//boost::numeric::ublas::vector<double>* MMSBpoisson::getVecH(){					// hessian of L w.r.t alpha
//	boost::numeric::ublas::vector<double>* vecH = new boost::numeric::ublas::vector<double>(K);
//	for (int k = 0; k < K; ++k) {
//		vecH->operator ()(k)= num_users*(utils->trigamma(alpha->operator ()(k)));
//	}
//	return vecH;
//}


void MMSBpoisson::initializeUserIndex(unordered_map<int,int>* userList){  
	for(std::unordered_map<int,int>::iterator it=userList->begin(); it!=userList->end(); it++){
		userIndexMap->insert({it->second, it->first});
	}	
}

void MMSBpoisson::initializeGamma(){
//	cout<<"In InitializeGamma\n";
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
	double gamma_pk=0;
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




double MMSBpoisson::getUniformRandom(){
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
	

	std::unordered_map<int,int>* userList = new std::unordered_map<int,int>();
	std::unordered_set<int>* threadList = new std::unordered_set<int>();
	std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<pair<int,int>>>* userAdjlist = 
		new std::unordered_map<std::pair<int,int>,std::unordered_map<int,int>*, class_hash<pair<int,int>>>();
	std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost = 
		new std::unordered_map<std::pair<int,int>,std::vector<int>*, class_hash<pair<int,int>>>();

//	username_mention_graph.txt

	utilsClass->readThreadStructureFile(argv[1], userList, threadList, userAdjlist, userThreadPost);

	std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<pair<int,int>>>* heldUserAdjlist = 
		new std::unordered_map<std::pair<int,int>,std::unordered_map<int,int>*, class_hash<pair<int,int>>>();

	utilsClass->getTheHeldoutSet(userAdjlist, heldUserAdjlist, 0.10);

//	testDataStructures(userList,threadList, userAdjlist,userThreadPost);

//	cout<<endl<<i<<" "<<INT_MAX<<endl;

//	cout<<"Before MMSB constructor"<<endl;
	MMSBpoisson* mmsb = new MMSBpoisson(utilsClass);

//	cout<<"after MMSB constructor call"<<endl;
//	mmsb->getParameters(matFile, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
	double stepSizeNu = atof(argv[6]);
	mmsb->initialize(K, userList, threadList, userAdjlist, heldUserAdjlist, userThreadPost, stepSizeNu);	
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
