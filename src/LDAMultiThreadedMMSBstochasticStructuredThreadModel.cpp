//============================================================================
// Name        : ThreadStructuredMMSBpoissonforForums.cpp
// Author      : Abhimanu Kumar
// Version     :
// Copyright   : Your copyright notice
// Description : MMSBpoisson in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <float.h>
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

#include <chrono>
#include <thread>

#include <unordered_map>
#include <unordered_set>

#include "Utils.h"
#include <math.h>

using namespace std;
using namespace boost::numeric::ublas;


//error: ‘sleep_for’ is not a member of ‘std::this_thread’ is solved via including -D_GLIBCXX_USE_NANOSLEEP
//#define _GLIBCXX_USE_NANOSLEEP

/*
 *
 * This gets the users in each thread of the forum.
 *
 */
std::unordered_map<int,std::unordered_set<int>*>* getPerThreadUserSet(std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<pair<int,int>>>* userAdjlist){
	std::unordered_map<int,std::unordered_set<int>*>* perThreadUserSet  = new std::unordered_map<int,std::unordered_set<int>*>();
	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>::iterator it1= userAdjlist->begin(); it1!=userAdjlist->end(); ++it1){
		if(perThreadUserSet->count(it1->first.second)<=0)
			perThreadUserSet->insert({it1->first.second, new std::unordered_set<int>()});
		perThreadUserSet->at(it1->first.second)->insert(it1->first.first);
	}
	return perThreadUserSet;
}

/*
 *
 * This method takes the userList which is a hashmap and contains <userId, index> pair and returns
 * a hashmap containing <index, userId> hashmap.
 *
 */

std::unordered_map<int,int>* initializeUserIndex(unordered_map<int,int>* userList){ 
	std::unordered_map<int,int>* userIndexMap = new unordered_map<int,int>();
	for(std::unordered_map<int,int>::iterator it=userList->begin(); it!=userList->end(); it++){
		userIndexMap->insert({it->second, it->first});
	}	
	return userIndexMap;
}

/*
 *
 * This will print any general matrix 
 *
 */

template <class T>
void printMat(matrix<T> *mat, int M, int N) {
	for (int k = 0; k < M; ++k) {
		for (int j = 0; j < N; ++j) {
			cout << (*mat)(k,j) << "," ;
		}
		cout << endl;
	}
}

/*
 *
 * Prints just the Log Likelihood in a file 
 *
 */

void printLogLLtoFile(std::vector<double>* heldLogLikelihood, std::string logLLfile){

	ofstream outfile(logLLfile);
	for(std::vector<double>::iterator it = heldLogLikelihood->begin(); it!=heldLogLikelihood->end(); ++it){
		outfile<<(*it)<<endl;
	}
}

/*
 *
 * Prints the Log Likelihood as well as time it takes to 
 * reach that likelihood in a file 
 *
 */

void printLogLLtoFile(std::vector<double>* heldLogLikelihood, std::vector<double>* globalTimeList, std::string logLLfile){

	ofstream outfile(logLLfile);
	int ind=0;
	for(std::vector<double>::iterator it = heldLogLikelihood->begin(); it!=heldLogLikelihood->end(); ++it){
		outfile<<(*it)<< ","<<globalTimeList->at(ind)/(60*1.0)<<endl;
		ind++;
	}
}

/*
 *
 * Prints the Pi of the model into a file 
 *
 */

template <class T>
void printPiToFile(matrix<T> *mat, int M, int N, std::string fileName, unordered_map<int,int>* userIndexMap){
	ofstream outfile(fileName);
	for (int k = 0; k < M; ++k) {
		int userId = userIndexMap->at(k);
		outfile << userId <<",";
		for (int j = 0; j < N; ++j) {
			outfile << (*mat)(k,j) << "," ;
		}
		outfile << endl;
	}
}

/*
 *
 * Prints the topUsers in a thread that deviate alot  
 * fromm their global topic proportion in the thread.
 *
 */

//template <class T>
void printPerUserThreadTopicStats(std::string fileName, std::vector<std::unordered_map<int,std::vector<double>*>*>* perUserThreadTopicStats_thread_list, int K, std::vector<std::unordered_map<int,int>*>* perUserTopicStats_thread_list, int num_users, std::unordered_map<int,int>* userIndexMap, int topicDiffTopKUsers, int numParallelThreads){
	ofstream outfile(fileName);
	std::unordered_map<int,int>* final_map = new std::unordered_map<int,int>();

	for(int u=0; u<num_users; ++u){
		final_map->insert({u,0});
		for(int thr=0; thr<numParallelThreads; ++thr){
			if(perUserTopicStats_thread_list->at(thr)->count(u)>0)
				final_map->at(u)+=perUserTopicStats_thread_list->at(thr)->at(u);
		}
	} 
	for(int ind=1; ind<10; ++ind){
		double countSum=0;
		for(int k=0; k<K; k++){
			double count = 0;
			for(int thr=0; thr<numParallelThreads; ++thr){
				count+=perUserThreadTopicStats_thread_list->at(thr)->at(ind)->at(k);
			}
			outfile<<count<<",";
			countSum+=count;
		}
		outfile<<countSum<<endl;
	}
	std::unordered_set<int>* keySetHigh = new std::unordered_set<int>();
	std::unordered_set<int>* keySetLow = new std::unordered_set<int>();
	for(int n=0; n<topicDiffTopKUsers; ++n){
		int currUserLow = 0;
		int currUserHigh = 0;
		for(std::unordered_map<int,int>::iterator it = final_map->begin(); it!=final_map->end(); ++it){
		   if(keySetLow->count(it->first)<=0 && it->second<final_map->at(currUserLow) && it->second>5*K)
			   currUserLow=it->first; 
		   if(keySetHigh->count(it->first)<=0 && it->second>final_map->at(currUserHigh))
			   currUserHigh=it->first;
		}
		keySetHigh->insert(currUserHigh);
		keySetLow->insert(currUserLow);
	}
	outfile<<"\n========Bottom "<<topicDiffTopKUsers<<"=======\n";
	for(std::unordered_set<int>::iterator it=keySetLow->begin(); it!=keySetLow->end(); ++it){
		outfile<<userIndexMap->at(*it)<<",";
	}
	outfile<<"\n========Top "<<topicDiffTopKUsers<<"=======\n";
	for(std::unordered_set<int>::iterator it=keySetHigh->begin(); it!=keySetHigh->end(); ++it){
		outfile<<userIndexMap->at(*it)<<",";
	}
	outfile<<endl;
    outfile.flush();
	outfile.close();
}

/*
 *
 * This prints the topics topKWords in each topic
 *
 */

void printLDATopics(matrix<double> *mat, int M, int N, std::string fileName, int topKWords, std::unordered_map<int, std::string>* vocabMap){
	ofstream outfile(fileName);
	for(int k=0; k<M; ++k){
		std::unordered_set<int>* keySet = new std::unordered_set<int>();
		outfile<<k+1;
		for(int n=0; n<topKWords; ++n){
			int currentWord = 0;
			for(int v=0; v<N; ++v){
				if(keySet->count(v)>0)
					continue;
				if((*mat)(k,v)>(*mat)(k,currentWord))
					currentWord=v;
			}
			keySet->insert(currentWord);
            outfile<<","<<vocabMap->at(currentWord);
		}                             
		outfile<<endl;
	}	
}

void printLDATopics(matrix<double> *mat, int M, int N, std::string fileName){
	ofstream outfile(fileName);
	double sumTopics[M];
	for(int k=0; k<M; ++k){
		sumTopics[k]=0;
		for(int v=0; v<N; ++v){
//			(*mat)(k,v)=rand()*1.0/((N+1)*RAND_MAX);
			sumTopics[k]+=(*mat)(k,v);
		}                             
	}	
	for(int k=0; k<M; ++k){
		outfile<<k+1;
		for(int v=0; v<N; ++v){
			outfile<<","<<(*mat)(k,v)/sumTopics[k];
		}                             
		outfile<<endl;
	}	
}

/*
 *
 * this prints any matrix to file
 *
 */

template <class T>
void printToFile(matrix<T> *mat, int M, int N, char* fileName) {
	ofstream outfile(fileName);
	for (int k = 0; k < M; ++k) {
		for (int j = 0; j < N; ++j) {
			outfile << (*mat)(k,j) << "," ;
		}
		outfile << endl;
	}
}

/*
 *
 * This is for debugging purposes and prints 
 * any matrix that has NAN element. It tells
 * which indices has NAN
 *
 */

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

/*
 *
 * This is for debugging purposes and prints 
 * any matrix that has negative element. It tells
 * which indices has negative value
 *
 */

template <class T>
void printNegInMat(matrix<T> *mat, int M, int N) {
	bool flag=false;
	cout<<"In printNegInMat\t";
	for (int k = 0; k < M; ++k) {
		for (int j = 0; j < N; ++j) {
			if(((*mat)(k,j))<=0){
				flag=true;
				cout << (*mat)(k,j) << ","<<k<<","<<j<<"||\t";
			}
		}
	}
	if(flag)
		cout << endl;
}

/*
 *
 * This is for debugging purposes and prints 
 * any matrix that has negative or NAN element. It tells
 * which indices has negative or NAN value
 *
 */

template <class T>
void printNegOrNanInMat(matrix<T> *mat, int M, int N) {
	bool flag=false;
	cout<<"In printNegOrNanInMat\t";
	for (int k = 0; k < M; ++k) {
		for (int j = 0; j < N; ++j) {
			if(((*mat)(k,j))<=0 || std::isnan((*mat)(k,j))){
				flag=true;
				cout << (*mat)(k,j) << ","<<k<<","<<j<<"||\t";
				exit(0); 
			}
		}
	}
	if(flag)
		cout << endl;
}

/*
 *
 * This is for debugging purposes and prints 
 * all the datastructures used for model 
 * computation
 *
 */

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

/*
 *
 * It prints a 3D matrix with the third dimension 
 * separated by ||==||
 *
 */

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

/*
 *
 * This give the prediction results for 
 * simple LDA model. The prediction results 
 * for the LDA just use topic discovered to 
 * predict the link weight 
 *
 */

double getBaseLinePrediction(std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* heldUserAdjlist_held, int inputCountOffset){
	double mean = 0, numEdges=0, basePrediction=0;
	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<pair<int,int>>>::iterator it1=heldUserAdjlist_held->begin(); it1!=heldUserAdjlist_held->end(); ++it1){
		for(std::unordered_map<int,int>::iterator it2 = it1->second->begin(); it2!=it1->second->end(); ++it2){
			int Y_pq = it2->second + inputCountOffset;
			numEdges++;
			mean+=Y_pq;
		}
	}
	mean = mean/numEdges;
	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<pair<int,int>>>::iterator it1=heldUserAdjlist_held->begin(); it1!=heldUserAdjlist_held->end(); ++it1){
		for(std::unordered_map<int,int>::iterator it2 = it1->second->begin(); it2!=it1->second->end(); ++it2){
			int Y_pq = it2->second + inputCountOffset;
			basePrediction += abs(Y_pq-mean);
		}
	}
	basePrediction = basePrediction/numEdges;
	return basePrediction;
}

class MMSBpoisson{

private:
	boost::numeric::ublas::vector<double>* alpha;
	boost::numeric::ublas::vector<double>* eta;
	matrix<double>* gamma;							// for MMSB
	matrix<double>* tau;							// for LDA
	unordered_map<int,int>* userIndexMap;			// map from userIndex to original user id.
    std::unordered_set<int>* threadList;                                                                           
    std::unordered_set<int>* vocabList;                                                                           
	std::unordered_map<int, std::string>* vocabMap;
    std::unordered_map<int,int>* userList;			// map from original userId to userIndex
	std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* userAdjlist; // stores by original user and thread ids
    std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost;       
    std::unordered_map<int,std::vector<int>*>* perThreadUserList;

    std::unordered_map< std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>* perUserThreadChiStats4Phi;
//	std::vector<std::unordered_map<int,int>*>* testThreadedMap;
	std::vector<std::unordered_map< std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>*>* perUserThreadChiStats4Phi_thread_list;
    std::unordered_map< std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>* perUserThreadPhiStats4Chi;
	std::vector<std::unordered_map< std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>*>* perUserThreadPhiStats4Chi_thread_list;
    std::unordered_map< std::pair<int,int>, int, class_hash<pair<int,int>>>* perUserThreadDelta;
    std::unordered_map<int, std::unordered_set<int>*>* perThreadUserSet;

	int inputCountOffset=0;
	double chi_epsilon = 0.05;//1.5;	//1e-7;					// chi_epsilon should be high else it gives inf in phi updates
	double link_epsilon = 1e-1;
	int threadPostLengthThreshold = 10000;//200;
    double textFactorForNWTextBalance = 1e-3;
	int topKWords = 50;//20;
	double diagHyperInit;
	double nonDiagHyperInit;
	std::string outputDir = "outputDir";
	char* outputFile;
	bool fasterFlag = false;
	int constantThreads=100;
	double nwFactorForText=1.0;
	bool printFullLDATopics=false;
	double topicDiffConstant=0.20;
	int topicDiffTopKUsers=20;

	int numParallelThreads;
	std::vector<std::thread>* parallelThreadList;
	std::vector<bool>* parallelComputationFlagList;
	std::vector<bool>* threadKillFlagList;
	// heldout set, also strores test set: 1 is test 0 is held
	std::unordered_map< std::pair<int,int>, std::unordered_map<int, std::pair<int,int>>*, class_hash<pair<int,int>>>* heldUserAdjlist;
	std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* heldUserAdjlist_held;        
	std::vector<std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>*>* heldUserAdjlist_thread_list;
	std::vector<std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>*>* testUserAdjlist_thread_list;
//	boost::multi_array<double,4>* phiPQ;
//	matrix<double>* B;

//	matrix<double>* held_phi_gh_sum;
//	matrix<double>* held_phi_y_gh_sum;
//	matrix<double>* held_phi_lgammaPhi;

	std::vector< matrix<double>*>* held_phi_gh_sum_thread_list;
	std::vector< matrix<double>*>* held_phi_pg_sum_thread_list;
	std::vector<matrix<double>*>* held_phi_qh_sum_thread_list;
	std::vector<matrix<double>*>* held_phi_y_gh_sum_thread_list;
	std::vector<matrix<double>*>* held_phi_lgammaPhi_thread_list;
	std::vector<matrix<double>*>* held_phi_gh_pq_thread_list;

	std::vector<double>* heldLLcomputation_thread_list;


	//Phi terms
	matrix<double>* phi_gh_sum;
	matrix<double>* phi_y_gh_sum;
	matrix<double>* phi_qh_sum;
	matrix<double>* phi_pg_sum;
	matrix<double>* phi_lgammaPhi;
	double phi_logPhi;

	matrix<double>* chi_kv_sum;
	
	std::vector<double>* prediction_error_thread_list;

	std::vector<matrix<double>*>* phi_gh_sum_thread_list;
	std::vector<matrix<double>*>* phi_y_gh_sum_thread_list;
	std::vector<matrix<double>*>* phi_qh_sum_thread_list;
	std::vector<matrix<double>*>* phi_pg_sum_thread_list;
	
	std::vector<matrix<double>*>* chi_kv_sum_thread_list;

	matrix<double>* nu;
	matrix<double>* lambda;
	matrix<double>* kappa;
	matrix<double>* theta;
	int num_users;
	int num_threads;
	int vocab_size;
	int numHeldoutEdges;
	int numHeldoutPosts=0;
	int numTotalLinks;
	int K;            
	int nuIter;
	double stepSizeNu=0;
	Utils* utils;
	matrix<double>* bDenomSum;

    int zeroEdgesTimes;
    char* seedIndexFileName;

	double stochasticSampleNodeMultiplier;
	double stochasticSamplePairMultiplier;
	std::vector<double>* multiThreadNetworkSampleSizeList;
	double multiThreadGlobalNetworkSampleSize;

	double stochasticSamplePostsMultiplier;
	std::vector<double>* multiThreadPostsSampleSizeList;
	double multiThreadGlobalPostsSampleSize;

//	matrix<int>* inputMat;
	double multiplier;
	static const  int variationalStepCount=10;
	static constexpr double threshold=1e-5;
	static constexpr double alphaStepSize=1e-6;
	static constexpr double stepSizeMultiplier=0.5;
	static constexpr double globalThreshold=1e-4;

	static constexpr double const_log_epsilon_delta=1.1;

	double stochastic_step_tau=1;
	double stochastic_step_kappa=2;//0.5;
	double stochastic_step_alpha=1.0;

    double samplingThreshold=0.5;
	double samplingThreadThreshold=0.2;

	double heldoutPredictionError = 0.0;
	std::vector<std::unordered_map<int,std::vector<double>*>*>* perUserThreadTopicStats_thread_list;
	std::vector<std::unordered_map<int,int>*>* perUserTopicStats_thread_list;

public:
	MMSBpoisson(Utils *);
	void getParameters(int iter_threshold, int inner_iter,int nu_iter);
	double getUniformRandom();
//	matrix<double>* updatePhiVariational(int p, int q, double sumGamma_p, double sumGamma_q);
	double getVariationalLogLikelihood();
//	void updateB(int p, int q, matrix<double>* oldPhi_pq);

	matrix<double>* multiThreadStochasticUpdateTau();
    
	matrix<double>* stochasticUpdateLambda();
	matrix<double>* stochasticUpdateNuFixedPoint();
	matrix<double>* stochasticUpdateGamma(int p, int q);
	void stochasticVariationalUpdatesPhi(int p, int q, int Y_pq, int thread_id, int Y_qp);
	double stochasticUpdateGlobalParams(int inner_iter, int* num_iters);
    void getPerThreadUserList();

	void updateB();
	void updateGamma(int p);
	void updateNu();
	void updateNuFixedPoint();
	void updateLambda();

	void setZeroEdges(int zeroEdges);
	void setHyperInits(double diagHyperInit, double nonDiagHyperInit);
	void setSeedIndexFileName(char* seedIndexFileName);
	void setVocabMap(std::unordered_map<int, std::string>* vocabMap);
	void setFasterFlag(bool fasterFlag);
	void setConstantThreads(int constantThreads);
	void setNwFactorForText(double nwFactorForText);
	void setPrintFullLDATopics(bool printFullLDATopics);

	double getStochasticStepSize(int iter_no);
	
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
	void initializeTau();
	void initializeAlpha();
	void initializeAlpha(double initValue);
	void initializeEta();
	void initialize(int K, std::unordered_map<int,int>* userList, 
	std::unordered_set<int>* threadList, std::unordered_set<int>* vocabList,
	std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* userAdjlist,
	std::unordered_map< std::pair<int,int>, std::unordered_map<int, std::pair<int,int>>*, class_hash<pair<int,int>>>* heldUserAdjlist,
	std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost,
	double stepSizeNu, int numHeldoutEdges, double stochastic_step_kappa, double samplingThreshold, int numParallelThreads, int vocabSize,
	std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* heldUserAdjlist_held);        
	
	matrix<double>* getMeanBlockMat();
	double getPredictionForEdge(matrix<double>* meanBlockMat, int p, int q);
	
	std::vector<double>* getPerThreadLDATopicVector(std::pair<int,int> user_thread, std::vector<double>* constDigamma, std::vector<double>* user_topic, matrix<double>* pi, int threadID);
	void printPerThreadLDAUserTopics(std::vector<int>* threadList_thread, int threadID);
	void performEndOfThreadTask(std::vector<int>* threadList_thread, int threadID);

	void initializeB();

	void initializeAllPhiMats();
	void initializeChiMats();
	void initializeChiPhiStatsOnce();
	void cleanUpChiPhiStats(int threadID);

	void initializeUserIndex(std::unordered_map<int,int>* userList);
	
	void initializeNu();
	void initializeLambda();
	void initializeTheta();
	void initializeKappa();
	
	double getHeldoutLogLikelihood();
	double getParallelHeldoutLL(int threadID);
	
	void getParametersInParallel(int iter_threshold, int inner_iter, int nu_iter, int stochastic_tau, 
			char* outputFile, std::unordered_map<int, std::unordered_set<int>*>* perThreadUserSet,
			int numTotalLinks, double textFactorForNWTextBalance);// 
	bool areThreadsComputing();
	void sendThreadKillSignal();
	void tellThreadsToCompute();
	double syncAndGlobalUpdate(int iter);
	void threadEntryFunction(std::vector<int>* threadList_thread, int threadID);

	boost::numeric::ublas::vector<double>* multiThreadStochasticUpdateGamma(int p);
	void multiThreadedStochasticVariationalUpdatesPhi(int p, int q, int Y_pq, int thread_id, int Y_qp,
			int threadID, pair<int,int> user_thread_p, pair<int,int> user_thread_q, int real_delta_tp);
	double multiThreadGlobalMatsFromLocal();
	void multiThreadStochasticUpdateGlobalParams(int iter);
	void multiThreadStochasticUpdateChi(int p, std::pair<int,int> user_thread, int real_delta_tp, int threadID, std::vector<double>* constDigamma);
	void initializeMultiThreadMats(std::vector<int>* threadList_thread, int threadID);
	void multiThreadParallelUpdate(std::vector<int>* threadList_thread, int threadID);
	matrix<double>* multiThreadStochasticUpdateNuFixedPoint();
	matrix<double>* multiThreadStochasticUpdateLambda();

	matrix<double>* getPis();
	boost::numeric::ublas::vector<double>* getVecH();
	boost::numeric::ublas::vector<double>* getVecG();

	// devugging
	std::vector<std::vector<double>*>* updateChiTextFactorVec; 
	std::vector<std::vector<double>*>* updateChiNetworkFactorVec; 
	int chiDebugParams = 4;
	int debugThreadID = 0;
};

MMSBpoisson::MMSBpoisson(Utils* utils){
//	cout<<"In MMSB constructor"<<endl;
	this->utils = utils;
}

/*
 *
 * The numbe rof zeroEdges to sample from each 
 * thread for stochastic variational inference 
 *
 */

void MMSBpoisson::setZeroEdges(int zeroEdges){
	this->zeroEdgesTimes = zeroEdges;
}

/*
 *
 * This sets the faster flag to compute the 
 * log likelihood. In case of faster flag on
 * we compute a shorter version of log likelihood
 *
 */

void MMSBpoisson::setFasterFlag(bool fasterFlag){
	this->fasterFlag = fasterFlag;
}

/*
 *
 * This sets the number of threads to sampled 
 * randomly in each iteration. 
 *
 */

void MMSBpoisson::setConstantThreads(int constantThreads){
	this->constantThreads=constantThreads;
}

/*
 *
 * This is the weight that text portion gets from
 * the n/w side
 *
 */

void MMSBpoisson::setNwFactorForText(double nwFactorForText){
	this->nwFactorForText=nwFactorForText;
}

/*
 *
 * This sets the flag for printng the full LDA 
 * topic probabilities
 *
 */

void MMSBpoisson::setPrintFullLDATopics(bool printFullLDATopics){
	this->printFullLDATopics=printFullLDATopics;
}

/*
 *
 * This sets the vocabMap which has the form 
 * <index, string> pair 
 *
 */

void MMSBpoisson::setVocabMap(std::unordered_map<int, std::string>* vocabMap){
	this->vocabMap = vocabMap;
}

/*
 *
 * This sets the hyper parameters of the block
 * matrix of MMSB 
 *
 */

void MMSBpoisson::setHyperInits(double diagHyperInit, double nonDiagHyperInit){
	this->diagHyperInit = diagHyperInit;
	this->nonDiagHyperInit = nonDiagHyperInit;
}

/*
 *
 * This sets the seedIndex file that is used
 * for intialization of the model for 
 * faster convergence 
 *
 */

void MMSBpoisson::setSeedIndexFileName(char* seedIndexFileName){
	this->seedIndexFileName = seedIndexFileName;
}

/*
 *
 * This intializes the alpha hyperparameter 
 * for the MMSB Pi parameter 
 *
 */

void MMSBpoisson::initializeAlpha(){
	for (int k = 0; k < K; ++k) {
		(*alpha)(k)= 0.01;//0.5+(getUniformRandom()-0.5)*0.1;
	}
}

void MMSBpoisson::initializeAlpha(double initValue){
	for (int k = 0; k < K; ++k) {
		(*alpha)(k)= initValue;//0.01;//0.5+(getUniformRandom()-0.5)*0.1;
	}
}

/*
 *
 * This intializes the eta hyperparameter 
 * for the MMSB Block matrix 
 *
 */

void MMSBpoisson::initializeEta(){
	for (int k = 0; k < vocab_size; ++k) {
		(*eta)(k)= 0.01;//(getUniformRandom())*0.5;
	}
//	cout<<"initialized Eta"<<endl;
}

/*
 *
 * This intializes the datastructure for the model.
 * This is the first function called after initializing
 * the MMSBPoisson class 
 *
 */

void MMSBpoisson::initialize(int K, std::unordered_map<int,int>* userList,
	std::unordered_set<int>* threadList, std::unordered_set<int>* vocabList, 
	std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* userAdjlist,
	std::unordered_map< std::pair<int,int>, std::unordered_map<int, std::pair<int,int>>*, class_hash<pair<int,int>>>* heldUserAdjlist,
	std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost,
	double stepSizeNu, int numHeldoutEdges, double stochastic_step_kappa, double samplingThreshold, int numParallelThreads, int vocabSize,
	std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* heldUserAdjlist_held){        

	this->threadList = threadList;
	this->vocabList = vocabList;
	this->userAdjlist = userAdjlist;
	this->heldUserAdjlist = heldUserAdjlist;
	this->heldUserAdjlist_held = heldUserAdjlist_held;
	this->userThreadPost = userThreadPost;
    this->userList = userList;

	this->num_users = userList->size();	// this stays the same even with heldout as we donot delete from original
	this->num_threads = threadList->size();
	this->vocab_size = vocabSize;//vocabList->size(); //there are some words that are not in the posts (starter)

	this->numParallelThreads = numParallelThreads;
	this->parallelThreadList = new std::vector<std::thread>(numParallelThreads);
	this->parallelComputationFlagList = new std::vector<bool>(numParallelThreads);
	this->threadKillFlagList = new std::vector<bool>(numParallelThreads);

//	for(int k=0;k<K;k++)cout<<(*alpha)(k)<<" ";
//	cout<<endl;
	cout<<"num_users:"<<num_users<<"; stepSizeNu: "<<stepSizeNu<<"; num_threads: "<<num_threads<<
		"; numHeldoutEdges: "<<numHeldoutEdges<<"; stochastic_step_kappa: "<<stochastic_step_kappa<<
		"; samplingThreshold/floatThreshold: "<<samplingThreshold<<"; numParallelThreads: "<<numParallelThreads<<
		"; vocab_size"<<vocab_size<<endl;
	this->K=K;            
	this->stepSizeNu=stepSizeNu;
	this->numHeldoutEdges = numHeldoutEdges;
	this->stochastic_step_kappa = stochastic_step_kappa;
	this->samplingThreshold = samplingThreshold;
	cout<<" (num_threads*num_users*num_users-numHeldoutEdges)"
		<<((double)num_threads*num_users*num_users-numHeldoutEdges)	<<endl;
//TODO: At present we are not using samplingThreshold
//TODO: At present it is used as floatThreshold
	
	
	stochasticSamplePostsMultiplier = ((double)num_threads*num_users-numHeldoutPosts);//*samplingThreshold);

	stochasticSampleNodeMultiplier = ((double)num_threads*num_users*num_users-numHeldoutEdges)/(2.0);//*samplingThreshold);
	stochasticSamplePairMultiplier = ((double)num_threads*num_users*num_users - numHeldoutEdges)/(2.0);//*samplingThreshold); 
	//div by 2 coz we update P-> and q<-p edges simultaneously; this is still slightly wrong because heldout just considers either p->q or q<-p which we have to rectify
	//
	cout<<"stochasticSamplePairMultiplier: "<<stochasticSamplePairMultiplier<<endl;
	
	gamma = new matrix<double>(num_users,K);
	tau = new matrix<double>(K,vocab_size);

	userIndexMap = new unordered_map<int,int>();

//	B = new matrix<double>(K,K);
	nu = new matrix<double>(K,K);
	lambda = new matrix<double>(K,K);
	kappa = new matrix<double>(K,K);
	theta = new matrix<double>(K,K);
	alpha = new boost::numeric::ublas::vector<double>(K);
	eta = new boost::numeric::ublas::vector<double>(vocab_size);

//	phiPQ = new boost::multi_array<double, 4>(boost::extents[K][K][num_users][num_users]);

//	held_phi_gh_sum = new matrix<double>(K,K);
//	held_phi_y_gh_sum = new matrix<double>(K,K);
//	held_phi_lgammaPhi = new matrix<double>(K,K);

	phi_gh_sum = new matrix<double>(K,K);
	phi_y_gh_sum = new matrix<double>(K,K);
	phi_qh_sum = new matrix<double>(num_users,K);
	phi_pg_sum = new matrix<double>(num_users,K);
	phi_lgammaPhi = new matrix<double>(K,K);
	phi_logPhi = 0;

    chi_kv_sum = new matrix<double>(K,vocab_size);

	initializeUserIndex(userList);

	getPerThreadUserList();

//  cout<< "Hello there!"<<endl;

	multiplier = alphaStepSize;
	//		this->inputMat = inputMat;
//	this->num_users = userList->size();//num_users;
//	cout<<"num_users "<<num_users<<endl;
//	this->K=K;
	initializeAlpha();					// dirichlet prior for MMSB
	initializeEta();					//dirichlet prior for LDA
	//		initializeB();

	initializeNu();
	initializeLambda();
	initializeKappa();
	initializeTheta();
	
	initializeGamma();
	initializeTau();

//	cout<<"After intializeGamma\n";
//	for(int k=0;k<K;k++)cout<<(*alpha)(k)<<" ";
//	cout<<endl;
//	printMat(gamma,num_users,K);
//	cout<<"Fag end of initialize()\n";
}

/*
 *
 * This give a list of users that 
 * have posted on thread, for every thread
 * present in the forum
 *
 */

void MMSBpoisson::getPerThreadUserList(){
	perThreadUserList  = new std::unordered_map<int,std::vector<int>*>();
	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>::iterator it1= userAdjlist->begin(); it1!=userAdjlist->end(); ++it1){
		if(perThreadUserList->count(it1->first.second)<=0)
			perThreadUserList->insert({it1->first.second, new std::vector<int>()});
		perThreadUserList->at(it1->first.second)->push_back(it1->first.first);
	}
}


matrix<double>* MMSBpoisson::getMeanBlockMat(){
	matrix<double>* meanBlockMat= new matrix<double>(K,K);
	for(int g =0; g<K; g++)
		for(int h=0; h<K; h++)
			(*meanBlockMat)(g,h) = (*nu)(g,h)*(*lambda)(g,h);
	return meanBlockMat;

}

double MMSBpoisson::getPredictionForEdge(matrix<double>* meanBlockMat, int p, int q){
	double pi_p_sum = 0, pi_q_sum=0, Y_predicted=0;
	for(int k=0; k<K; k++){
		pi_p_sum+=(*gamma)(p,k);
        pi_q_sum+=(*gamma)(q,k);
	}
	for(int g=0; g<K; g++)
		for(int h=0; h<K; h++)
			Y_predicted += ((*meanBlockMat)(g,h)*((*gamma)(p,g)/pi_p_sum)*((*gamma)(q,h)/pi_q_sum));	

	return Y_predicted;
}

double MMSBpoisson::getParallelHeldoutLL(int threadID){

	double ll=0;
	double phi_sum = 0;
	clock_t begin = clock();
	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
			(*held_phi_y_gh_sum_thread_list->at(threadID))(g,h)=0;
			(*held_phi_gh_sum_thread_list->at(threadID))(g,h)=0;
			(*held_phi_lgammaPhi_thread_list->at(threadID))(g,h) =0;
		}
		for(int u=0; u<num_users; ++u){
			(*held_phi_pg_sum_thread_list->at(threadID))(u,g)=0;
			(*held_phi_qh_sum_thread_list->at(threadID))(u,g)=0;
		}
	}
	double held_phi_logPhi = 0;
	
	clock_t end = clock();

	matrix<double>* meanBlockMat = getMeanBlockMat();
	prediction_error_thread_list->at(threadID) = 0;
//	if((end-begin)/CLOCKS_PER_SEC > 0)
//		cout<<"LL first K^2 loop: "<< (end-begin)/CLOCKS_PER_SEC<<";\n";

//	matrix<double> phi_gh_pq(K,K);// = new matrix<double>(K,K);
	int totalEdges = 0;

//	std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<pair<int,int>>>* adjList;
//	std::vector<int>* threadDebugList = new std::vector<int>(); 

//	if(threadID==numParallelThreads){
//		for(int i=0; i<numParallelThreads; ++i)
//			threadDebugList->push_back(i);
////		adjList = heldUserAdjlist;
//	}
//	else
//		threadDebugList->push_back(threadID);
////		adjList = heldUserAdjlist_thread_list->at(threadID);

//	int originalThreadID = threadID;

	begin = clock();

	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<pair<int,int>>>::iterator it1=heldUserAdjlist_thread_list->at(threadID)->begin(); it1!=heldUserAdjlist_thread_list->at(threadID)->end(); ++it1){
		for(std::unordered_map<int,int>::iterator it2 = it1->second->begin(); it2!=it1->second->end(); ++it2){
//			cout<< "In heldoutLog-Likeli\n";
			int p = userList->at(it1->first.first);
			int q = userList->at(it2->first);
//			cout<< "In heldoutLog-Likelii after p,q index access\n";
			double digamma_p_sum = getDigamaValue(getMatrixRowSum(gamma,p,K));
			double digamma_q_sum = getDigamaValue(getMatrixRowSum(gamma,q,K));
			int Y_pq = it2->second + inputCountOffset;
			double Y_predicted = getPredictionForEdge(meanBlockMat,p,q);
			prediction_error_thread_list->at(threadID) += (abs(Y_pq-Y_predicted));
//			cout<<"Y_pq, Y_predicted: "<<Y_pq<<", "<<Y_predicted<<endl<<flush;
			totalEdges++;
			phi_sum=0;
			for(int g=0;g<K;g++){
				for(int h=0;h<K;h++){
					(*held_phi_gh_pq_thread_list->at(threadID))(g,h) = exp(dataFunctionPhiUpdates(g,h,Y_pq) 
							+ (getDigamaValue((*gamma)(p,g)) - digamma_p_sum)
							+ (getDigamaValue((*gamma)(q,h)) - digamma_q_sum));
					phi_sum += (*held_phi_gh_pq_thread_list->at(threadID))(g,h);
					if(std::isnan((*held_phi_gh_pq_thread_list->at(threadID))(g,h))){
						cout<<"dataFunc "<<dataFunctionPhiUpdates(g,h,Y_pq)<<"; gamma_pg, gamma_ph "<<
							(*gamma)(p,g)<<","<<(*gamma)(q,h)<<"; digamma_p, digamma_q "<<digamma_p_sum
							<<","<<digamma_q_sum<<endl;
						exit(0);
					}
//					ll+=(Y_pq*log((*lambda)(g,h)) - (*lambda)(g,h)-lgamma(Y_pq+1));
				}
			}
			for(int g=0;g<K;g++){
				for(int h=0;h<K;h++){
					if(phi_sum==0){
                        (*held_phi_gh_pq_thread_list->at(threadID))(g,h)=1.0/(K*K*1.0);
//						cout<<" HARD BUG TO CATCH\n";
					}else
						(*held_phi_gh_pq_thread_list->at(threadID))(g,h) = (*held_phi_gh_pq_thread_list->at(threadID))(g,h)/phi_sum;          
					if((*held_phi_gh_pq_thread_list->at(threadID))(g,h)!=0)
						held_phi_logPhi += ((*held_phi_gh_pq_thread_list->at(threadID))(g,h)*log((*held_phi_gh_pq_thread_list->at(threadID))(g,h)));
					(*held_phi_gh_sum_thread_list->at(threadID))(g,h)+=(*held_phi_gh_pq_thread_list->at(threadID))(g,h);
					(*held_phi_y_gh_sum_thread_list->at(threadID))(g,h)+=((*held_phi_gh_pq_thread_list->at(threadID))(g,h)*Y_pq);
					(*held_phi_lgammaPhi_thread_list->at(threadID))(g,h) += ((*held_phi_gh_pq_thread_list->at(threadID))(g,h)*lgamma(Y_pq+1));//(*inputMat)(p,q)+1));

					(*held_phi_pg_sum_thread_list->at(threadID))(p,g) += (*held_phi_gh_pq_thread_list->at(threadID))(g,h);
					(*held_phi_qh_sum_thread_list->at(threadID))(q,h) += (*held_phi_gh_pq_thread_list->at(threadID))(g,h);
				}
			}
		}
	}


	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
			ll+=((*held_phi_y_gh_sum_thread_list->at(threadID))(g,h)*(log((*lambda)(g,h)) + getDigamaValue((*nu)(g,h))) 
				-(*lambda)(g,h)*(*nu)(g,h)*(*held_phi_gh_sum_thread_list->at(threadID))(g,h) - (*held_phi_lgammaPhi_thread_list->at(threadID))(g,h));
		}
	}

	// The following calculation is not part of heldoutLogLikelihood() function
	// Also it is likelihood for the variational parameters

	if(!fasterFlag){
		for(int u=0; u<num_users; u++){
			double held_digamma_sum = 0;
			for(int g=0; g<K; g++)
				held_digamma_sum += (*gamma)(u,g);
			held_digamma_sum = getDigamaValue(held_digamma_sum);
			for(int g=0; g<K; g++){
				ll+=((*held_phi_pg_sum_thread_list->at(threadID))(u,g)*(getDigamaValue((*gamma)(u,g)) - held_digamma_sum));
				ll+=((*held_phi_qh_sum_thread_list->at(threadID))(u,g)*(getDigamaValue((*gamma)(u,g)) - held_digamma_sum));
			}
		}
		ll-=held_phi_logPhi;
	}
		
	end = clock();
//	if((end-begin)/CLOCKS_PER_SEC > 0)
//		cout<<"LL: "<<ll<<" bigger loop: "<< (end-begin)/CLOCKS_PER_SEC<<"; totalEdges "<<totalEdges<<" in THREAD "<<threadID<<"\n";


	return ll;
}

double MMSBpoisson::getHeldoutLogLikelihood(){
	double ll=0;
	double phi_sum = 0;
	matrix<double>* held_phi_gh_sum = new matrix<double>(K,K);
	matrix<double>* held_phi_y_gh_sum = new matrix<double>(K,K);
	matrix<double>* held_phi_lgammaPhi = new matrix<double>(K,K);
	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
			(*held_phi_y_gh_sum)(g,h)=0;
			(*held_phi_gh_sum)(g,h)=0;
			(*held_phi_lgammaPhi)(g,h) =0;
		}
	}
	
	matrix<double>* phi_gh_pq = new matrix<double>(K,K);
	
	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<pair<int,int>>>::iterator it1=heldUserAdjlist_held->begin(); it1!=heldUserAdjlist_held->end(); ++it1){
		for(std::unordered_map<int,int>::iterator it2 = it1->second->begin(); it2!=it1->second->end(); ++it2){
//			cout<< "In heldoutLog-Likeli\n";
			int p = userList->at(it1->first.first);
			int q = userList->at(it2->first);
//			cout<< "In heldoutLog-Likelii after p,q index access\n";
			double digamma_p_sum = getDigamaValue(getMatrixRowSum(gamma,p,K));
			double digamma_q_sum = getDigamaValue(getMatrixRowSum(gamma,q,K));
			int Y_pq = it2->second + inputCountOffset;
			phi_sum=0;
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
//	cout<< "digamma value "<<value<<endl;
//	try{
//    boost::math::digamma(value);
//	}catch(...){
//	cout<< "digamma for value "<<value<<endl;

//	}
//	cout<<"after digamma value" << endl;
	if(value<DBL_MIN)
		value=DBL_MIN*10;
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
	int num_iters = 0;


//	int iter_threshold = 30;
//	int inner_iter = 4;
	do{
        iter++;
		cout<<"iter "<<iter<<endl;
		oldLL=newLL;
//		newLL=updateGlobalParams(inner_iter);
		newLL=stochasticUpdateGlobalParams(inner_iter, &num_iters);
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

void MMSBpoisson::getParametersInParallel(int iter_threshold, int inner_iter, int nu_iter, 
		int stochastic_tau, char* outputFile, std::unordered_map<int, std::unordered_set<int>*>* perThreadUserSet,
		int numTotalLinks, double textFactorForNWTextBalance){ 
	this->numTotalLinks = numTotalLinks;
	// TODO: change phi coz it is K*K*N*N 
//	initialize(num_users, K);//, inputMat);			// should be called from the main function
	cout<<"iter_threshold: "<<iter_threshold<<"; inner_iter: "<<inner_iter<<"; nu_iter: "<<nu_iter
		<<"; stochastic_tau: "<<stochastic_tau<<"; outputFile: "<<outputFile<<"; perThreadUserSet "<<
		perThreadUserSet->size()<<"; numTotalLinks "<<numTotalLinks<<"; textFactorForNWTextBalance: "<<textFactorForNWTextBalance
		<<"; zeroEdgesTimes: "<<zeroEdgesTimes<<"; constantThreads: "<<constantThreads<<"; fasterFlag: "<<fasterFlag
		<<"; nwFactorForText: "<<nwFactorForText<<endl;
	for(int k=0;k<K;k++)cout<<(*alpha)(k)<<" ";
	cout<<endl;
//	cout<<"ll-0"<<getVariationalLogLikelihood()<<endl;
//	boost::numeric::ublas::vector<double>* oldAlpha = new boost::numeric::ublas::vector<double>(K);
//	copyAlpha(oldAlpha);
	double newLL = 0;//getVariationalLogLikelihood();
	this->outputFile = outputFile;
	this->numTotalLinks = numTotalLinks;
	this->textFactorForNWTextBalance = textFactorForNWTextBalance; 
	double oldLL = 0;
	int iter=0;
	this->nuIter = nu_iter;
	int num_iters = 0;

    this->perThreadUserSet = perThreadUserSet;

	this->stochastic_step_tau = stochastic_tau;


	std::vector<double>* heldLogLikelihood = new std::vector<double>();

//	matrix<double>* phi_gh_sum;
//	matrix<double>* phi_y_gh_sum;
//	matrix<double>* phi_qh_sum;
//	matrix<double>* phi_pg_sum;

    //set variables and spawn the threads

    initializeChiPhiStatsOnce();

	//TODO: We dont need a global phi4chi

	perUserThreadChiStats4Phi_thread_list = new std::vector<std::unordered_map< std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>*>(numParallelThreads);
	perUserThreadPhiStats4Chi_thread_list = new std::vector<std::unordered_map< std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>*>(numParallelThreads);

	perUserThreadTopicStats_thread_list = new std::vector<std::unordered_map<int,std::vector<double>*>*>(numParallelThreads);
	perUserTopicStats_thread_list = new std::vector<std::unordered_map<int,int>*>(numParallelThreads);
//	std::vector<std::unordered_map<int,int>*>* 
	
	prediction_error_thread_list = new std::vector<double> (numParallelThreads);

	phi_gh_sum_thread_list = new std::vector<matrix<double>*>(numParallelThreads);
	phi_y_gh_sum_thread_list = new std::vector<matrix<double>*>(numParallelThreads);
	phi_qh_sum_thread_list = new std::vector<matrix<double>*>(numParallelThreads);
	phi_pg_sum_thread_list = new std::vector<matrix<double>*>(numParallelThreads);

	held_phi_gh_sum_thread_list = new std::vector<matrix<double>*>(numParallelThreads);
	held_phi_pg_sum_thread_list = new std::vector<matrix<double>*>(numParallelThreads);
	held_phi_qh_sum_thread_list = new std::vector<matrix<double>*>(numParallelThreads); ;
	held_phi_y_gh_sum_thread_list = new std::vector<matrix<double>*>(numParallelThreads); ;
	held_phi_lgammaPhi_thread_list = new std::vector<matrix<double>*>(numParallelThreads); ;
	held_phi_gh_pq_thread_list = new std::vector<matrix<double>*>(numParallelThreads); 

	heldUserAdjlist_thread_list = new std::vector<std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>*>(numParallelThreads);
	heldLLcomputation_thread_list = new std::vector<double>(numParallelThreads);

	chi_kv_sum_thread_list = new std::vector<matrix<double>*>(numParallelThreads);

	multiThreadNetworkSampleSizeList = new std::vector<double>(numParallelThreads);
	
	multiThreadPostsSampleSizeList = new std::vector<double>(numParallelThreads);

	// debugging
    updateChiTextFactorVec = new std::vector<std::vector<double>*>(numParallelThreads);
    updateChiNetworkFactorVec = new std::vector<std::vector<double>*>(numParallelThreads);

	int perThread_threadNum = num_threads*1.0/numParallelThreads;
//	int perThreadindex = 0;

	for(int i_threads=0; i_threads<numParallelThreads; ++i_threads){
		heldUserAdjlist_thread_list->at(i_threads) = new std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>();
	}

	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>::iterator it=heldUserAdjlist_held->begin(); it!=heldUserAdjlist_held->end(); ){
		for(int i_threads=0; i_threads<numParallelThreads; i_threads++){
			heldUserAdjlist_thread_list->at(i_threads)->insert({it->first,it->second});
			++it;
			if(it==heldUserAdjlist_held->end())
				break;
		}

		if(it==heldUserAdjlist_held->end())
			break;
	
	}
	
	std::unordered_set<int>::iterator it = threadList->begin();

	for(int i_threads=0; i_threads<numParallelThreads; i_threads++){
		std::vector<int>* threadList_thread = new std::vector<int>();
		
		for(int indx=0; indx<perThread_threadNum; indx++){
			if(it==threadList->end())
				break;
			threadList_thread->push_back(*it);
			it++;
		}

        //this is for the case when 
		while(i_threads==numParallelThreads-1 && it!=threadList->end()){	//this is for the case when 
			threadList_thread->push_back(*it);
			it++;
		}

		this->parallelComputationFlagList->at(i_threads)=false;
		this->threadKillFlagList->at(i_threads)=false;

        perUserThreadPhiStats4Chi_thread_list->at(i_threads) = new std::unordered_map<std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>();
        perUserThreadChiStats4Phi_thread_list->at(i_threads) = new std::unordered_map<std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>();

		perUserThreadTopicStats_thread_list->at(i_threads) = new std::unordered_map<int, std::vector<double>*>();
		perUserTopicStats_thread_list->at(i_threads) = new std::unordered_map<int, int>();


		phi_gh_sum_thread_list->at(i_threads) = new matrix<double>(K,K);
		phi_y_gh_sum_thread_list->at(i_threads) = new matrix<double>(K,K);
		phi_qh_sum_thread_list->at(i_threads) = new matrix<double>(num_users,K);
		phi_pg_sum_thread_list->at(i_threads) = new matrix<double>(num_users,K);
		
		chi_kv_sum_thread_list->at(i_threads) = new matrix<double>(K, vocab_size);

		held_phi_gh_sum_thread_list->at(i_threads) = new matrix<double>(K,K);  
		held_phi_pg_sum_thread_list->at(i_threads) = new matrix<double>(num_users,K);  
		held_phi_qh_sum_thread_list->at(i_threads) = new matrix<double>(num_users,K); 
		held_phi_y_gh_sum_thread_list->at(i_threads) = new matrix<double>(K,K); 
		held_phi_lgammaPhi_thread_list->at(i_threads) = new matrix<double>(K,K); 
		held_phi_gh_pq_thread_list->at(i_threads) = new matrix<double>(K,K);  

		//debugging
		updateChiTextFactorVec->at(i_threads) = new std::vector<double>(chiDebugParams);//mean,number, max, min
		updateChiNetworkFactorVec->at(i_threads) = new std::vector<double>(chiDebugParams);//mean,number, max, min


		parallelThreadList->at(i_threads) = std::thread(&MMSBpoisson::threadEntryFunction, this, threadList_thread, i_threads); 
	}

	iter_threshold = iter_threshold*inner_iter;	// this is for efficent calculation of stochasticStepSize
	std::chrono::seconds main_second(1);

    int stabilityNum=0;
	double floatThreshold = samplingThreshold;//5;//1e-3;


	//start main computation in an inf loop.
	
	int consec_dec_ll = 0;
	int iterThreshLL = 1;
	
    double baselineHeld=getBaseLinePrediction(heldUserAdjlist_held, inputCountOffset);
	time_t start, end;
	double globalTime = 0;
	std::vector<double>* globalTimeList = new std::vector<double>();
	do{
        time(&start);
        if(iter!=0 && !areThreadsComputing()){				//if threads are not working then dpo the main computation
			debugThreadID = rand()%numParallelThreads;
			oldLL=newLL;
//		newLL=updateGlobalParams(inner_iter);
			clock_t begin = clock();
			newLL = syncAndGlobalUpdate(iter);					//main computation
			heldLogLikelihood->push_back(newLL);
			cout<< "THREAD LL "<<newLL<<endl;
			clock_t end = clock();
			if((end-begin)/CLOCKS_PER_SEC > 0)
				cout<<"syncAndGlobalUpdate: "<< (end-begin)/CLOCKS_PER_SEC<<";\t";
//			if(iter%iterThreshLL == 0){
			begin = clock();
//			newLL = getHeldoutLogLikelihood();
			if (newLL<oldLL)
				consec_dec_ll++;
//				stochastic_step_alpha*=1.5;
			else if(newLL>oldLL)
				consec_dec_ll=0;
//				stochastic_step_alpha/=1.5;
			if(consec_dec_ll>=4/iterThreshLL){
				stochastic_step_alpha*=(2*iterThreshLL);
				consec_dec_ll=0;
			}
			end = clock();
//			}
			if((end-begin)/CLOCKS_PER_SEC > 0)
				cout<<"LogLikelihood calculation: "<< (end-begin)/CLOCKS_PER_SEC<<";\t";
			cout<<"iter "<<iter<<" held-LL"<<newLL<< " floatThresh "<<abs((newLL-oldLL)/newLL)<<"\n"<<flush;
			cout<<"prediction error: "<<heldoutPredictionError/numHeldoutEdges<<"\n"<<flush;
			cout<<"baseline prediction error: "<<baselineHeld<<"\n"<<flush;
			cout<<"debugThreadID: "<<debugThreadID<<endl;
			if(abs(newLL-oldLL)<floatThreshold)
				stabilityNum++;
			else
				stabilityNum=0;
		}
		
		tellThreadsToCompute();

		while(areThreadsComputing()){						// main sleeps and let the threads compute
			std::this_thread::sleep_for(main_second);
		}

		iter++;
		 
		//		newLL=stochasticUpdateGlobalParams(inner_iter, &num_iters);

		// TODO flag up the threads to start their computation 
		// (NOTE this can be made more efficient by starting the threads before the LL computation by main)


		if(iter>=iter_threshold || stabilityNum==6){
			sendThreadKillSignal();
			break;
		}
		time(&end);
		double diff = difftime(end,start);
		globalTime+=diff;
		globalTimeList->push_back(globalTime);
		cout<<"GLOBAL TIME DIFF "<< diff<<" "<<globalTime;
	}while(1);//abs(oldLL-newLL)>globalThreshold);

	//Flag all the threads to end their computation
	cout<<"\nWaiting for thread joins\n";
	
    for(int i=0; i<numParallelThreads; i++)
		parallelThreadList->at(i).join();// join threads

	matrix<double>* pi = getPis();
//	cout<<"Gamma\n";
//	printMat(gamma,num_users,K);
//	cout<<"Nu\n";
//	printMat(nu,K,K);
//	cout<<"Lambda\n";
//	printMat(lambda,K,K);
//	cout<<"PI\n";
//	printMat(pi,num_users,K);
	
	std::ostringstream s;
	s<<outputDir<<"/"<<outputFile;
	printPiToFile(pi,num_users,K,s.str(),userIndexMap);	
	if(textFactorForNWTextBalance>0){
		std::ostringstream s1;
		s1<<outputDir<<"/"<<outputFile<<".perTopicTop"<<topKWords<<"Words.txt";
		if(printFullLDATopics){
			std::ostringstream sf;
			sf<<outputDir<<"/"<<outputFile<<".perTopicFullWordIndices.txt";
			printLDATopics(tau, K, vocab_size, sf.str());
		}
//		printLDATopics(tau, K, vocab_size, s1.str(), topKWords, vocabMap);
//		TODO: change the top K words to this as well?
		printLDATopics(chi_kv_sum, K, vocab_size, s1.str(), topKWords, vocabMap);
		std::ostringstream s2;
		s2<<outputDir<<"/"<<outputFile<<".perUserThreadTopicStats.txt";
		// TODO: debug this
		printPerUserThreadTopicStats(s2.str(), perUserThreadTopicStats_thread_list, K, perUserTopicStats_thread_list, num_users, userIndexMap, topicDiffTopKUsers, numParallelThreads);
	}
//	printErrorToFile(prediction_error_thread_list, num_threads);
	std::ostringstream logLLfile;
	logLLfile<<outputDir<<"/"<<outputFile<<".LL.txt";
//	printLogLLtoFile(heldLogLikelihood, logLLfile.str());
	printLogLLtoFile(heldLogLikelihood, globalTimeList, logLLfile.str());


}


void MMSBpoisson::printPerThreadLDAUserTopics(std::vector<int>* threadList_thread, int threadID){
	matrix<double>* pi = getPis();
	std::ostringstream s;
	s<<outputDir<<"/"<<outputFile<<".perUserThreadTopic_thread_"<<threadID<<".txt";
	ofstream outfile(s.str()); 
	std::vector<double>* constDigamma = new std::vector<double>(K);
	for(int ind=1;ind<10;ind++){
		perUserThreadTopicStats_thread_list->at(threadID)->insert({ind,new std::vector<double>(K)});
	}

	for(int k=0; k<K; k++){
		for(int v=0; v<vocab_size; v++){
			if(v==0)
				constDigamma->at(k) = (*tau)(k,v);
			else
				constDigamma->at(k) += (*tau)(k,v);
		}
		for(int ind=1;ind<10;ind++){
			perUserThreadTopicStats_thread_list->at(threadID)->at(ind)->at(k)=0;
		}
	}
	
	for(std::vector<int>::iterator it = threadList_thread->begin(); it!=threadList_thread->end(); ++it){
		int curr_thread = (*it);
		for(std::vector<int>::iterator it2 = perThreadUserList->at(curr_thread)->begin(); it2!=perThreadUserList->at(curr_thread)->end(); ++it2){
			int curr_user = (*it2);
			std::pair<int,int> user_thread = std::make_pair(curr_user, curr_thread);
			std::vector<double>* user_topic = new std::vector<double>(K);
			user_topic = getPerThreadLDATopicVector(user_thread, constDigamma, user_topic, pi, threadID);
			outfile<<curr_user<<","<<curr_thread;
			for(int k=0; k<K; k++){
				outfile<<","<<user_topic->at(k);
			}
			outfile<<endl;
		}
	}
	outfile.flush();
	delete pi;
}

std::vector<double>* MMSBpoisson::getPerThreadLDATopicVector(std::pair<int,int> user_thread, std::vector<double>* constDigamma, std::vector<double>* user_topic, matrix<double>* pi, int threadID){
	double samplingPostWordsThreshold = threadPostLengthThreshold*1.0/(userThreadPost->at(user_thread)->size()*1.0);
	double multiplyingFactor = (samplingPostWordsThreshold<1)? 1.0/samplingPostWordsThreshold:1;
	double chi_sum = 0;
	int p = userList->at(user_thread.first);
	std::vector<double>* chi_tpi = new std::vector<double>(K);
	double chiStats_p =0, delta_tp=0, log_epsilon_delta=0;
	if(perUserThreadDelta->count(user_thread)>0)
		delta_tp = perUserThreadDelta->at(user_thread);
	if(delta_tp>0){
		log_epsilon_delta = chi_epsilon/(1.0*delta_tp);
		if(log_epsilon_delta<1)
			log_epsilon_delta = const_log_epsilon_delta;				//TODO: make it a constant
	}
	for(int k=0; k<K; k++){
		user_topic->at(k)=0;
	}
	double topic_sum =0;

	for(std::vector<int>::iterator it = userThreadPost->at(user_thread)->begin(); it!=userThreadPost->at(user_thread)->end(); ++it){
		double randGenerated = rand()*1.0/RAND_MAX;//getUniformRandom(); // this was throwing seg fault
		//		cout<<"randGenerated: " << randGenerated<<" postSize: "<<userThreadPost->at(user_thread)->size()<<endl;
		if(randGenerated>samplingPostWordsThreshold){
			//			cout<<"Skipping since the post is long" << endl;
			continue;
		}
		//		cout<<"Not Skipping " << endl;

		int wordId = (*it);
		if(wordId >= vocab_size)
			continue;
		if(wordId >= vocab_size)
			cout<<"wordId vocab_size "<<wordId << " "<<vocab_size<<endl;
		chi_sum=0;
		bool infFalg =false;
		for(int k=0; k<K; k++){
			//if(real_delta_tp>0)
			//	updatesFromPhi = log(log_epsilon_delta)*(1 - phiStatsForChi_p->at(k)) + 
			//		phiStatsForChi_p->at(k)*log(1 + log_epsilon_delta);
			double localUpdatesFromPhi = nwFactorForText* log(log_epsilon_delta)*(1 - (*pi)(p,k)) + ((*pi)(p,k))*log(1 + log_epsilon_delta);
			chi_tpi->at(k) = exp(getDigamaValue((*tau)(k,wordId)) - getDigamaValue(constDigamma->at(k)) + localUpdatesFromPhi);
			if(std::isnan(chi_tpi->at(k))||chi_tpi->at(k)<0){
				cout<<"in multiThreadStochasticUpdateChi; chi_tpi->at(k): "<<chi_tpi->at(k)<<"; "<<(*tau)(k,wordId)<<"; "<<constDigamma->at(k)<<endl;//<<user_thread.first<<","<<user_thread.second<<endl;
				exit(0);
			}
			if(std::isinf(chi_tpi->at(k)))
				infFalg=true;
			chi_sum +=chi_tpi->at(k);
		}
		//		if(chi_sum<=DBL_MIN)
		for(int k=0; k<K; k++){
			if(chi_sum<=DBL_MIN || infFalg)
				chi_tpi->at(k) = 1.0/(1.0*K);
			else
				chi_tpi->at(k) = chi_tpi->at(k)/chi_sum;
			user_topic->at(k) += multiplyingFactor*chi_tpi->at(k);
			topic_sum += multiplyingFactor*chi_tpi->at(k);
			if(std::isnan(chi_tpi->at(k))||chi_tpi->at(k)<0){
				cout<<"in Normalization multiThreadStochasticUpdateChi; chi_tpi->at(k): "<<chi_sum<<"; "<<chi_tpi->at(k)<<"; "<<(*tau)(k,wordId)<<"; "<<constDigamma->at(k)<<endl;//<<user_thread.first<<","<<user_thread.second<<endl;
				exit(0);
			}
			//TODO: put in updates for phiStats4Chi
		}
		infFalg=false;

		//		for()
	}
	for(int k=0; k<K; k++){
		if(topic_sum<DBL_MIN)
			user_topic->at(k)= 1.0/K;
		else
			user_topic->at(k)= user_topic->at(k)/topic_sum;
		double topicDiff = abs((*pi)(p,k) - user_topic->at(k));
		for(int ind=1;ind<10;ind++){
			if(topicDiff*10 > ind){
				perUserThreadTopicStats_thread_list->at(threadID)->at(ind)->at(k) +=1;
			}
		}
		if(topicDiff>=topicDiffConstant){
			if(perUserTopicStats_thread_list->at(threadID)->count(p)>0){
				perUserTopicStats_thread_list->at(threadID)->at(p) +=1;
			}else{
				perUserTopicStats_thread_list->at(threadID)->insert({p,1});
			}
		}
	}
	return user_topic;
}

void MMSBpoisson::sendThreadKillSignal(){
	for(int i=0; i<numParallelThreads; i++)
		threadKillFlagList->at(i) = true;
}

/*
 * This methods return true when threads are doing computatiom
 * and false when main is doing computation
 *
 */

void MMSBpoisson::tellThreadsToCompute(){
	for(int i=0; i<numParallelThreads; i++)
		parallelComputationFlagList->at(i) = true;
}

bool MMSBpoisson::areThreadsComputing(){
    bool flag = false;
	for(int i=0; i<numParallelThreads; i++){
		flag = flag || parallelComputationFlagList->at(i);
	}
	return flag;
}

double MMSBpoisson::syncAndGlobalUpdate(int iter){
	clock_t begin = clock();
    double ll = multiThreadGlobalMatsFromLocal();
	clock_t end = clock();
	cout<<"From Main Thread, computing multiThreadGlobalMatsFromLocal time:"<<(begin-end)/CLOCKS_PER_SEC<<endl;
	begin = clock();
	multiThreadStochasticUpdateGlobalParams(iter);	
	end = clock();
	cout<<"From Main Thread, computing multiThreadStochasticUpdateGlobalParams time:"<<(begin-end)/CLOCKS_PER_SEC<<endl;
	return ll;
}

double MMSBpoisson::multiThreadGlobalMatsFromLocal(){
//	cout<<"Start of method multiThreadGlobalMatsFromLocal"<<endl;
	initializeAllPhiMats();			// this sets all the global phi_mats to 0
	if(textFactorForNWTextBalance>0){
		initializeChiMats();
	}
	multiThreadGlobalNetworkSampleSize = 0;
	multiThreadGlobalPostsSampleSize = 0;
	double ll=0;
	heldoutPredictionError = 0;
	for(int thr=0; thr<numParallelThreads; thr++){
		for(int j=0; j<K; j++){
			for(int i=0; i<num_users; i++){
				(*phi_pg_sum)(i,j)+=(*phi_pg_sum_thread_list->at(thr))(i,j);
				(*phi_qh_sum)(i,j)+=(*phi_qh_sum_thread_list->at(thr))(i,j);
			}
			for(int i=0; i<K; i++){
				(*phi_gh_sum)(i,j)+=(*phi_gh_sum_thread_list->at(thr))(i,j);
				(*phi_y_gh_sum)(i,j)+=(*phi_y_gh_sum_thread_list->at(thr))(i,j);
			}
			if(textFactorForNWTextBalance>0){
				for(int v=0; v<vocab_size; ++v){
					(*chi_kv_sum)(j,v)+=(*chi_kv_sum_thread_list->at(thr))(j,v);
				}
			}
		}
		multiThreadGlobalNetworkSampleSize += multiThreadNetworkSampleSizeList->at(thr);
		multiThreadGlobalPostsSampleSize += multiThreadPostsSampleSizeList->at(thr);
		ll+=heldLLcomputation_thread_list->at(thr);
		heldoutPredictionError += prediction_error_thread_list->at(thr);
	}


	clock_t begin = clock();
	if(textFactorForNWTextBalance>0){
		for(int thr=0; thr<numParallelThreads; thr++){
			for(std::unordered_map< std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>::iterator it = perUserThreadChiStats4Phi_thread_list->at(thr)->begin(); it!=perUserThreadChiStats4Phi_thread_list->at(thr)->end(); ++it){
				for(int k=0; k<K; k++){
					perUserThreadChiStats4Phi->at(it->first)->at(k) = it->second->at(k);
					// This works on the assumption that on user_thread goes to exactly one process thread.
				}
			}
		}
	}
	// The followinf code takes to much time, bad code
//	clock_t begin = clock();
//	for(std::unordered_map< std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>::iterator it = perUserThreadChiStats4Phi->begin(); it!=perUserThreadChiStats4Phi->end(); ++it){
//		for(int k=0; k<K; k++){
//			it->second->at(k) = 0;
//			for(int thr=0; thr<numParallelThreads; thr++){
//				if(perUserThreadChiStats4Phi_thread_list->at(thr)->count(it->first)>0){
//					it->second->at(k) +=perUserThreadChiStats4Phi_thread_list->at(thr)->at(it->first)->at(k);
//				}
//			}
//		}
//	}
	clock_t end = clock();
//	cout<<"From Main Thread, computing perUserThreadChiStats4Phi in multiThreadGlobalMatsFromLocal time:"<<(begin-end)/CLOCKS_PER_SEC<<endl;

//	cout<<"end of method multiThreadGlobalMatsFromLocal"<<endl;
	return ll;
	// TODO: sync ChiPhi stats as well
	// NOTE: not updating perUserThreadPhiStats4Chi_thread_list 
//	cout<<"multiThreadGlobalNetworkSampleSize "<<multiThreadGlobalNetworkSampleSize<<";\t";
}

void MMSBpoisson::multiThreadStochasticUpdateGlobalParams(int iter){
//	cout<<"Start of method multiThreadStochasticUpdateGlobalParams"<<endl;

	double stochastic_step_size = getStochasticStepSize(iter);

	if(multiThreadGlobalNetworkSampleSize==0){
		cout<<"multiThreadGlobalNetworkSampleSize is 0\n";
		return;
	}
	if(multiThreadGlobalPostsSampleSize==0 && textFactorForNWTextBalance>0){
		cout<<"multiThreadGlobalPostsSampleSize is 0\n";
		exit(0);
	}

	for(int p=0; p<num_users; p++){
		boost::numeric::ublas::vector<double>* gamma_p = multiThreadStochasticUpdateGamma(p);
		for(int k=0; k<K; k++){
			if((*gamma)(p,k)<=0){
				cout<<"gamma before outer update "<<p<<" "<<k<<" "<<(*gamma)(p,k)<< endl;
			}
			(*gamma)(p,k)=((1-stochastic_step_size)*(*gamma)(p,k) + stochastic_step_size*(*gamma_p)(k));
			if((*gamma)(p,k)<=0){
				cout<<"In outer gamma update "<<p<<" "<<k<<" "<<(*gamma)(p,k)<<" "<<(*gamma_p)(k)<< endl;
				exit(0);
			}
		}
		delete gamma_p;
	}
//	cout<<"gamma\n";
//	printNegOrNanInMat(gamma,num_users,K);
	
	matrix<double>* nu_p = multiThreadStochasticUpdateNuFixedPoint();
	for(int l=0; l<K; l++)
		for(int m=0; m<K; m++)
			(*nu)(l,m) = ((1-stochastic_step_size)*(*nu)(l,m) + stochastic_step_size*(*nu_p)(l,m));
	delete nu_p;
//	printNegOrNanInMat(nu,K,K);
//	printMat(nu,K,K);

	matrix<double>* lambda_p = multiThreadStochasticUpdateLambda();
	for(int l=0; l<K; l++)
		for(int m=0; m<K; m++)
			(*lambda)(l,m) = ((1-stochastic_step_size)*(*lambda)(l,m) + stochastic_step_size*(*lambda_p)(l,m));
	delete lambda_p;
//	printNegOrNanInMat(lambda,K,K);

	if(textFactorForNWTextBalance>0){
		matrix<double>* tau_p = multiThreadStochasticUpdateTau();
		for(int k=0; k<K; k++){
			double sum_v=0;
			for(int v=0; v<vocab_size; v++){
				(*tau)(k,v) = ((1-stochastic_step_size)*(*tau)(k,v) + stochastic_step_size*(*tau_p)(k,v));
				sum_v+=(*tau)(k,v);
			}
//TODO: beta not tau needs tobe normalized
//			for(int v=0; v<vocab_size; v++){
//				(*tau)(k,v) = (*tau)(k,v)/sum_v;
//			}
		}
		delete tau_p;
		//TODO: tau needs to be normalized.
		cout<<"Tau Prints"<<endl;
//		printNegOrNanInMat(tau,K,vocab_size);
	} else{
		cout<<endl<<"textFactorForNWTextBalance is ZERO; thus LDA code not run"<<endl;
	}
//	cout<<"End of method multiThreadStochasticUpdateGlobalParams"<<endl;

}

/*
 * Main function where the local thread starts
 * */

void MMSBpoisson::threadEntryFunction(std::vector<int>* threadList_thread, int threadID){
	std::chrono::seconds thread_second(1);

	while(1){
//		cout<<"\t In thread "<<threadID<<";\t";
		while(!parallelComputationFlagList->at(threadID)){	// thread sleeps and let the main compute and set flag
			if(threadKillFlagList->at(threadID))
				break;							//kill yourself i.e. just exit the loop and join main
			std::this_thread::sleep_for(thread_second);
		}
		if(threadKillFlagList->at(threadID)){
			performEndOfThreadTask(threadList_thread, threadID);
			break;							//kill yourself i.e. just exit the loop and join main
		}

		clock_t begin = clock();
        double ll = getParallelHeldoutLL(threadID);
		clock_t end = clock();
//		cout<<"getParallelHeldoutLL time threadID: "<<threadID<<", "<<(end-begin)/CLOCKS_PER_SEC<<";\t";
		
		heldLLcomputation_thread_list->at(threadID) = ll;

//		cout<<"computed held-ll "<<ll<<" thread-"<<threadID;

		begin = clock();
		initializeMultiThreadMats(threadList_thread, threadID);
		end = clock();
//		if((end-begin)/CLOCKS_PER_SEC > 0)
//			cout<<"initializing Local Phis threadID: "<<threadID<<", "<<(end-begin)/CLOCKS_PER_SEC<<";\t";
		begin = clock();
		multiThreadParallelUpdate(threadList_thread, threadID);
		end = clock();
//		if((end-begin)/CLOCKS_PER_SEC > 0)
//			cout<<"parallel update Local Phis threadID: "<<threadID<<", "<< (end-begin)/CLOCKS_PER_SEC<<";\t";
		parallelComputationFlagList->at(threadID)=false;

		if(threadKillFlagList->at(threadID)){
			performEndOfThreadTask(threadList_thread, threadID);
			break;							//kill yourself i.e. just exit the loop and join main
		}
	}
	cout<<"exited thread "<<threadID<<endl;		
}

void MMSBpoisson::performEndOfThreadTask(std::vector<int>* threadList_thread, int threadID){
	cout<<"In performEndOfThreadTask"<<endl;
	if(textFactorForNWTextBalance>0)
		printPerThreadLDAUserTopics(threadList_thread, threadID);
}

void MMSBpoisson::initializeMultiThreadMats(std::vector<int>* threadList_thread, int threadID){
	// initialize Phi and Chi mats
	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
			(*phi_gh_sum_thread_list->at(threadID))(g,h) = 0;
			(*phi_y_gh_sum_thread_list->at(threadID))(g,h)=0;
		}
		for(int p=0; p<num_users; p++){
			(*phi_pg_sum_thread_list->at(threadID))(p,g) = 0;
			(*phi_qh_sum_thread_list->at(threadID))(p,g) = 0;
		}
		if(textFactorForNWTextBalance>0){
			for(int v=0; v<vocab_size; ++v){
				(*chi_kv_sum_thread_list->at(threadID))(g,v) = 0;
			}
		}

	}
}

void MMSBpoisson::cleanUpChiPhiStats(int threadID){
	//TODO: We dont need a global phi4chi

    perUserThreadChiStats4Phi_thread_list->at(threadID)->clear();
	perUserThreadPhiStats4Chi_thread_list->at(threadID)->clear();

//    cout<<"In cleanUpChiPhiStats\n";

//	if(perUserThreadPhiStats4Chi_thread_list->size()>0){
//		cout<<"Hello my size is "<<perUserThreadPhiStats4Chi_thread_list->size()<<endl;
//	std::unordered_map< std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>* phi4ChiMap =  perUserThreadPhiStats4Chi_thread_list->at(threadID);
////	}
//	std::unordered_map< std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>* chi4PhiMap =  perUserThreadChiStats4Phi_thread_list->at(threadID);
//
//	for(std::unordered_map< std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>::iterator it=phi4ChiMap->begin(); it!=phi4ChiMap->end(); ++it){
//
//        std::unordered_map< std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>::iterator eraseIt = it;
//		phi4ChiMap->erase(eraseIt);
////		if(it->second != NULL){
////			cout<<" deleted in phi4ChiMap "<<it->second<<"; ";
////			delete it->second;
////		}
////		cout<<endl<<endl;
////		phi4ChiMap->erase(it);
//	}
//	
//	for(std::unordered_map< std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>::iterator it=chi4PhiMap->begin(); it!=chi4PhiMap->end(); ++it){
//        std::unordered_map< std::pair<int,int>, std::vector<double>*,class_hash<pair<int,int>>>::iterator eraseIt = it;
//		chi4PhiMap->erase(eraseIt);
////		if(it->second != NULL){
////			cout<<" deleted in chi4PhiMap "<<it->second<<"; ";
////			delete it->second;
////
////		}
////		cout<<endl<<endl;
////		chi4PhiMap->erase(it);
//	}
}


void MMSBpoisson::multiThreadParallelUpdate(std::vector<int>* threadList_thread, int threadID){
//	cout<<"Start of method multiThreadParallelUpdate threadID "<<threadID<<endl;
	multiThreadNetworkSampleSizeList->at(threadID)=0;
	multiThreadPostsSampleSizeList->at(threadID) = 0;
	int localThreadNum = threadList_thread->size();
//	cout<<"localThreadNum: "<<localThreadNum<<"\t";
//	int constantThreads=100;									// TODO:  change this and mke it an argv
	int constantUsers=1000;
//	int zeroEdgesTimes = 0;
	double lda_time =0, poisson_time=0;
	clock_t begin_lda, end_lda, begin_poisson, end_poisson ;

	std::vector<double>* constDigamma = new std::vector<double>(K);
	if(textFactorForNWTextBalance>0){
		for(int k=0; k<K; k++){
			for(int v=0; v<vocab_size; v++){
				if(v==0)
					constDigamma->at(k) = (*tau)(k,v);
				else
					constDigamma->at(k) += (*tau)(k,v);
			}
		}
	}

//	cout<< "calling cleanUpChiPhiStats"<<endl;
	cleanUpChiPhiStats(threadID);

//	std::unordered_set<int>* tempThreadSet = new std::unordered_set<int>();
	int temp_constantThreads = (constantThreads>localThreadNum)? localThreadNum:constantThreads;
	for(int thr=0; thr<temp_constantThreads; thr++){
		int i_thr = rand()%localThreadNum;
		int curr_thread_id = threadList_thread->at(i_thr);
		//		tempThreadSet->insert(threadList_thread->at(i_thr));
		//	}
		//	for(std::vector<int>::iterator it=tempThreadSet->begin(); it!=tempThreadSet->end(); it++){
		//	for(std::vector<int>::iterator it=threadList_thread->begin(); it!=threadList_thread->end(); it++){
		//   	if(getUniformRandom()>samplingThreadThreshold)
		//			continue;

		//TODO: Do not traverse this set very computation intensive;	
		// select set of users form every thread
		//		std::unordered_set<int>* tempThreadUsers = new std::unordered_set<int>();
		int numThreadUser = perThreadUserList->at(curr_thread_id)->size();
		constantUsers = (constantUsers>numThreadUser)? numThreadUser:constantUsers;
//		if(constantUsers>1)
//			cout<<"\nthreadID: "<<threadID<<" constantUsers: "<<constantUsers<<";\n";
//	cout<<"End of method multiThreadParallelUpdate threadID "<<threadID<<endl;
		for(int users=0; users<constantUsers; ++users){
			int i_user = rand()%numThreadUser;
//			cout<<"numThreadUser "<<numThreadUser<<"; i_user "<<i_user<<"\t";
			int userid_p = perThreadUserList->at(curr_thread_id)->at(i_user);
			int p = userList->at(userid_p);
			pair<int,int> user_thread = std::make_pair(userid_p,curr_thread_id);

//			std::unordered_set<int>* pNeighbors = new std::unordered_set<int>();
			std::unordered_map<int,std::pair<int,int>>* pNeighborsYpq=new std::unordered_map<int,std::pair<int,int>>();
			int real_delta_tp=0;
            // Initialize the phiStats4Chi
			std::vector<double>* phiStats4Chi = new std::vector<double>(K);
			for(int k=0; k<K; k++)
				phiStats4Chi->at(k)=0;
//			cout<< "I am going to access; hopefully no seg-faults "<<endl;
//			cout<< "accessing "<<perUserThreadPhiStats4Chi_thread_list->at(threadID)<<"; ";
			perUserThreadPhiStats4Chi_thread_list->at(threadID)->insert({user_thread, phiStats4Chi});
			// NOTE: we can pull this off just by local variables because we update chi after phi
//			cout<< "accessed huzzah"<<endl;
            begin_poisson = clock();

			for(std::unordered_map<int,int>::iterator it=userAdjlist->at(user_thread)->begin(); it!=userAdjlist->at(user_thread)->end(); ++it){
				int userid_q = it->first;
				pair<int,int> user_thread_q = std::make_pair(userid_q,curr_thread_id);
				// modify the below 2 if statements to include delta requirements for chi
				if(heldUserAdjlist->count(user_thread)>0 && heldUserAdjlist->at(user_thread)->count(userid_q)>0)
					continue;
				if (heldUserAdjlist->count(user_thread_q)>0 && heldUserAdjlist->at(user_thread_q)->count(userid_p)>0)
					continue;                                      // We dont even include th pair
//				pNeighbors->insert(userid_q);
                
				int Y_pq = it->second;
				if(Y_pq>0) 
					real_delta_tp++;

				int Y_qp = (userAdjlist->count(user_thread_q)>0 && userAdjlist->at(user_thread_q)->count(userid_p)>0)?
					userAdjlist->at(user_thread_q)->count(userid_p):0;
				pNeighborsYpq->insert({userid_q, std::make_pair(Y_pq,Y_qp)});

				int zeroIter = 0;

				while(zeroIter<zeroEdgesTimes){
					int randomIndex = rand()%num_users;
				   	int randomUserId = userIndexMap->at(randomIndex);
					std::pair<int,int> rand_thread = std::make_pair(randomUserId, curr_thread_id);
					if(userAdjlist->at(user_thread)->count(randomUserId)>0)
						continue;
					if(perThreadUserSet->at(curr_thread_id)->count(randomUserId)>0)
						continue;
					if(heldUserAdjlist->count(rand_thread)>0 && heldUserAdjlist->at(rand_thread)->count(userid_p)>0)
						continue;
					if(heldUserAdjlist->count(user_thread)>0 && heldUserAdjlist->at(user_thread)->count(randomUserId)>0)
						continue;
					pNeighborsYpq->insert({randomUserId, std::make_pair(0,0)});
					zeroIter++;
					//cout<<"\t\t\t\t In the zero EDGE PRODUCTION\n"<<flush;
				}

			}

			// initialize phiStats4Chi
//			for(int k=0; k<K; k++){
//				phiStats4Chi
//			}

//			for(std::unordered_map<int,int>::iterator it=userAdjlist->at(user_thread)->begin(); it!=userAdjlist->at(user_thread)->end(); ++it){
			for(std::unordered_map<int,std::pair<int,int>>::iterator it=pNeighborsYpq->begin(); it!=pNeighborsYpq->end(); ++it){
				int userid_q = it->first;

				pair<int,int> user_thread_q = std::make_pair(userid_q,curr_thread_id);

				// add the new p to chiphi
				int q = userList->at(userid_q);
				if(p==q)
					continue;
//				pair<int,int> user_thread_q = std::make_pair(userid_q,curr_thread_id);
				int Y_pq = it->second.first, Y_qp=it->second.second;
				try{
					multiThreadedStochasticVariationalUpdatesPhi(p,q,Y_pq,curr_thread_id,Y_qp,threadID, 
							user_thread, user_thread_q, real_delta_tp);
					// we have to take pair

					multiThreadNetworkSampleSizeList->at(threadID)=multiThreadNetworkSampleSizeList->at(threadID)+1;
					//just increment by one; we have taken pair coputation already into account

				}catch( const std::exception &exc ){
					//					cout<<"In Exception:: threadID: "<<threadID<<"; localThreadNum: "<<localThreadNum<<"; multiThreadNetworkSampleSizeList: "<<multiThreadNetworkSampleSizeList->at(threadID)<<";\t";
					std::cerr << exc.what();
					cout<<"Exception in StochasticVariationUpdatePhi "<<endl;
					//					exit(0);
				}
			}
//			delete pNeighbors;
			delete pNeighborsYpq;
			end_poisson = clock();
			poisson_time += (end_poisson-begin_poisson);
			
			begin_lda = clock();
			if(textFactorForNWTextBalance>0){
				multiThreadPostsSampleSizeList->at(threadID) = multiThreadPostsSampleSizeList->at(threadID)+1;
				multiThreadStochasticUpdateChi(p, user_thread, real_delta_tp, threadID, constDigamma);
			}
            end_lda = clock();
            lda_time += (begin_lda-end_lda);
			// check to see if the above loop throws caught exception
		}
		//		delete tempThreadUsers;
}
//	cout<<"threadID: "<<threadID<<"; localThreadNum: "<<localThreadNum<<"; multiThreadNetworkSampleSizeList: "<<multiThreadNetworkSampleSizeList->at(threadID)<<";\t";
	cout<<"threadID: "<<threadID<<"; poisson_time: "<<poisson_time/CLOCKS_PER_SEC<<"; lda_time: "<<lda_time/CLOCKS_PER_SEC<<"forum_hreads: "<<temp_constantThreads<<"constantUsers: "<<constantUsers<<";\t";
//	cout<<"End of method multiThreadParallelUpdate threadID "<<threadID<<endl;
//delete tempThreadSet;
	delete constDigamma;
}

void MMSBpoisson::multiThreadStochasticUpdateChi(int p, std::pair<int,int> user_thread, int real_delta_tp, 
		int threadID, std::vector<double>* constDigamma){
	// TODO we have to update stats here for chiStats4Phi
	//
//	cout<<"Start of method multiThreadStochasticUpdateChi, threadID "<<threadID<<endl;
	if(userThreadPost->count(user_thread)<=0){
//		cout<<"End of method multiThreadStochasticUpdateChi as postLength=0, threadID "<<threadID<<endl;
		return;						
	}
	// return is a good idea for 2 reasons 1) no updating/updated, 2) phi is updated via global datastructure 


	std::vector<double>* chiUpdatesForTau_p = new std::vector<double>(K);
	std::vector<double>* chiUpdatesForPhi_p = new std::vector<double>(K);
	std::vector<long double>* chi_tpi = new std::vector<long double>(K);
	std::vector<double>* updatesFromPhi = new std::vector<double>(K);

	std::vector<double>* chiStats4Phi = new std::vector<double>(K);		// dont delete

	long double  chi_sum=0; 
	double log_epsilon_delta=0;
	log_epsilon_delta = chi_epsilon/(1.0*real_delta_tp);
//TODO Do we need const_log_epsilon_delta?
//	if(log_epsilon_delta<1)
//		log_epsilon_delta = const_log_epsilon_delta;

	std::vector<double>* phiStatsForChi_p =perUserThreadPhiStats4Chi_thread_list->at(threadID)->at(user_thread);
	// NOTE: above we are readiong from the local thread data-structure.

	for(int k=0; k<K; k++){
//		for(int v=0; v<vocab_size; v++){
//			if(v==0)
//				constDigamma->at(k) = (*tau)(k,v);
//			else
//				constDigamma->at(k) += (*tau)(k,v);
//		}
		chiUpdatesForPhi_p->at(k)=0;
		chiUpdatesForTau_p->at(k)=0;
		chiStats4Phi->at(k)=0;
		updatesFromPhi->at(k)=0;
		if(real_delta_tp>0)
			updatesFromPhi->at(k) = real_delta_tp*(log(log_epsilon_delta)*(1 - phiStatsForChi_p->at(k)) 
				+ phiStatsForChi_p->at(k)*log(1 + log_epsilon_delta));
//			updatesFromPhi->at(k) = (log(log_epsilon_delta)*(1 - phiStatsForChi_p->at(k)) 
//				+ phiStatsForChi_p->at(k)*log(1 + log_epsilon_delta);
	}
//	cout<<"In multiThreadStochasticUpdateChi "<<endl;
//	cout<<" going to access threadPost vector "<<userThreadPost->at(user_thread)<<endl;

	perUserThreadChiStats4Phi_thread_list->at(threadID)->insert({user_thread, chiStats4Phi});

	double samplingPostWordsThreshold = threadPostLengthThreshold*1.0/(userThreadPost->at(user_thread)->size()*1.0);
	double multiplyingFactor = (samplingPostWordsThreshold<1)? 1.0/samplingPostWordsThreshold:1;

//	if(samplingPostWordsThreshold <1)
//		cout<<"user_thread: " << user_thread.first<<", "<<user_thread.second<<" postSize: "<<userThreadPost->at(user_thread)->size()<<endl;

	// TODO: I commented out the continue code for dealing with longer posts
	multiplyingFactor = 1.0;

	bool debugFlagPrint = false;

	for(std::vector<int>::iterator it = userThreadPost->at(user_thread)->begin(); it!=userThreadPost->at(user_thread)->end(); ++it){
		// TODO: I commented out the continue code for dealing with longer posts
//		double randGenerated = rand()*1.0/RAND_MAX;//getUniformRandom(); // this was throwing seg fault
////		cout<<"randGenerated: " << randGenerated<<" postSize: "<<userThreadPost->at(user_thread)->size()<<endl;
//		if(randGenerated>samplingPostWordsThreshold){
////			cout<<"Skipping since the post is long" << endl;
//			continue;
//		}
//		cout<<"Not Skipping " << endl;

//		if(debugThreadID==threadID && real_delta_tp>0)
//			if((double)rand()/RAND_MAX < 0.01)
//				debugFlagPrint=true;
//			else
//				debugFlagPrint=false;
//		else
			debugFlagPrint=false;
		
		int wordId = (*it);
		if(wordId >= vocab_size)
			continue;
		if(wordId >= vocab_size)
			cout<<"wordId vocab_size "<<wordId << " "<<vocab_size<<endl;
		chi_sum=0;
		bool infFalg =false;
//		cout<<endl<<"phi_tp";
		for(int k=0; k<K; k++){
			//if(real_delta_tp>0)
			//	updatesFromPhi = log(log_epsilon_delta)*(1 - phiStatsForChi_p->at(k)) + 
			//		phiStatsForChi_p->at(k)*log(1 + log_epsilon_delta);
//			double textFactor = (getDigamaValue((*tau)(k,wordId)) - getDigamaValue(constDigamma->at(k)))
			chi_tpi->at(k) = exp((getDigamaValue((*tau)(k,wordId)) - getDigamaValue(constDigamma->at(k)))*1.0/nwFactorForText + updatesFromPhi->at(k));
			//chi_tpi->at(k) = exp(getDigamaValue((*tau)(k,wordId)) - getDigamaValue(constDigamma->at(k)) + nwFactorForText*updatesFromPhi->at(k));
			if(std::isnan(chi_tpi->at(k))||chi_tpi->at(k)<0){
				cout<<"in multiThreadStochasticUpdateChi; chi_tpi->at(k): "<<chi_tpi->at(k)<<"; "<<phiStatsForChi_p->at(k)<<"; "<<real_delta_tp<<"; "<<threadID<<"; "<<(*tau)(k,wordId)<<"; "<<updatesFromPhi->at(k)<<"; "<<constDigamma->at(k)<<";"<<getDigamaValue((*tau)(k,wordId)) - getDigamaValue(constDigamma->at(k))<<"; "<<exp(nwFactorForText*updatesFromPhi->at(k))<<nwFactorForText*updatesFromPhi->at(k)<<endl;//<<user_thread.first<<","<<user_thread.second<<endl;
				exit(0);
			}
			if(std::isinf(chi_tpi->at(k)))
				infFalg=true;
			chi_sum +=chi_tpi->at(k);
//			cout<<","<<k<<","<<perUserThreadPhiStats4Chi_thread_list->at(threadID)->at(user_thread)->at(k);
		}
//		if(chi_sum<=DBL_MIN)
		//if(chi_sum<=DBL_MIN || infFalg || std::isinf(chi_sum)){
		if(chi_sum<=DBL_MIN ){
			chi_sum=0;
			for(int k=0; k<K; k++){
				chi_tpi->at(k) = rand()*1.0/(K*RAND_MAX);
				chi_sum +=chi_tpi->at(k);
			}
		}
		if(debugFlagPrint)
			cout<<endl<<"chi_tpi/phi_tpi";
		double chi_tpi_sum=0;
		for(int k=0; k<K; k++){
			chi_tpi->at(k) = chi_tpi->at(k)/chi_sum;
			(*chi_kv_sum_thread_list->at(threadID))(k,wordId) += multiplyingFactor*(double)chi_tpi->at(k);
			chiStats4Phi->at(k) += multiplyingFactor*(double)chi_tpi->at(k);
			if(std::isnan(chi_tpi->at(k))||chi_tpi->at(k)<0){
				cout<<"in Normalization multiThreadStochasticUpdateChi; chi_tpi->at(k): "<<chi_sum<<"; "<<chi_tpi->at(k)<<"; "<<phiStatsForChi_p->at(k)<<"; "<<real_delta_tp<<"; "<<threadID<<"; "<<(*tau)(k,wordId)<<"; "<<updatesFromPhi->at(k)<<"; "<<constDigamma->at(k)<<endl;//<<user_thread.first<<","<<user_thread.second<<endl;
				exit(0);
			}
			//TODO: put in updates for phiStats4Chi
			if(debugFlagPrint){
				cout<<","<<k<<","<<perUserThreadPhiStats4Chi_thread_list->at(threadID)->at(user_thread)->at(k)/chi_tpi->at(k);
				chi_tpi_sum+=perUserThreadPhiStats4Chi_thread_list->at(threadID)->at(user_thread)->at(k)/(chi_tpi->at(k)*K);
			}
		}
		infFalg=false;
		if(debugFlagPrint){
			double chi_tpi_var=0;
			for(int k=0; k<K; k++){  
				chi_tpi_var += (chi_tpi->at(k)-chi_tpi_sum)*(chi_tpi->at(k)-chi_tpi_sum)/K;
			}
			cout<<endl<<"variance= "<<chi_tpi_var;
		}
		if(debugFlagPrint)
			cout<<endl;

//		for()
	}

	delete chiUpdatesForPhi_p;
	delete chiUpdatesForTau_p;
	delete chi_tpi;
	delete updatesFromPhi;
//	cout<<"End of method multiThreadStochasticUpdateChi, threadID "<<threadID<<endl;
}


double MMSBpoisson::getStochasticStepSize(int iter_no){
//double stochastic_step_tau=1;
//double stochastic_step_kappa=0.5;
double iter_step_size =  1.0/(pow(iter_no+stochastic_step_tau,stochastic_step_kappa)*stochastic_step_alpha);
//cout<<"iter_step_size "<<iter_step_size<<endl;
return iter_step_size;
}

double MMSBpoisson::stochasticUpdateGlobalParams(int inner_iter, int* num_iters){//(gamma,B,alpha,Y,inner_iter){
	double ll=0;
//	int nuIters=3;
	for(int i=1; i<=inner_iter;i++){
//		cout<<"i "<<i<<endl;
//		initializeAllPhiMats();							//this is not needed for stochastic updates
		double stochastic_step_size = getStochasticStepSize(*num_iters);
		(*num_iters)++;
		int num_nonzeros =0;
		for(int p=0; p<num_users; p++){
			for(unordered_set<int>::iterator it=threadList->begin(); it!=threadList->end(); it++){
				for(int q=p+1; q<num_users; q++){
					double randomNum = getUniformRandom();
//					cout<<"randomNum "<<randomNum<<endl;
					if(randomNum>samplingThreshold)	// TODO: this is outdateed code; using samplingThreshold as floatThreshold
						continue;
					int userid_p = userIndexMap->at(p);
//					cout<<"userid_p "<<p<<"\n";
					pair<int,int> user_thread = std::make_pair(userIndexMap->at(p),*it);
//					cout<<"userid_q "<<q<<"\n";
					int userid_q = userIndexMap->at(q);
//					cout<<"userid_q "<<userid_q<<"\n";
					//TODO also check whether it is in heldout
					if(p==q || (heldUserAdjlist->count(user_thread)>0 
								&& heldUserAdjlist->at(user_thread)->count(userid_q)>0) )
						continue;
					int Y_pq=0, Y_qp=0;
					if(userAdjlist->count(user_thread)>0){
						if(userAdjlist->at(user_thread)->count(userid_q)>0){ 
							Y_pq=userAdjlist->at(user_thread)->at(userid_q);
							pair<int,int> user_thread_q = std::make_pair(userid_q,*it);

							if(userAdjlist->count(user_thread_q)>0)
								if(userAdjlist->at(user_thread_q)->count(userid_p)>0)
									Y_qp = userAdjlist->at(user_thread_q)->at(userid_p);
							//cout<<"Y_pq "<<Y_pq<<"("<<user_thread.first<<","<<userid_q<<","<<user_thread.second<<")"<<"; ";
//							num_nonzeros++;
						}
					}
					try{
						stochasticVariationalUpdatesPhi(p,q,Y_pq,*it,Y_qp);//
					}catch(...){
						cout<<"Exception in StochasticVariationUpdatePhi "<<endl;
					}
					matrix<double>* gamma_pq=stochasticUpdateGamma(p,q);
					for(int k=0; k<K; k++){
						(*gamma)(p,k)=((1-stochastic_step_size)*(*gamma)(p,k) + stochastic_step_size*(*gamma_pq)(0,k));
						(*gamma)(q,k)=((1-stochastic_step_size)*(*gamma)(q,k) + stochastic_step_size*(*gamma_pq)(1,k));
					}
                    delete gamma_pq;

					matrix<double>* nu_p = stochasticUpdateNuFixedPoint();
					for(int l=0; l<K; l++)
						for(int m=0; m<K; m++)
                            (*nu)(l,m) = ((1-stochastic_step_size)*(*nu)(l,m) + stochastic_step_size*(*nu_p)(l,m));
					delete nu_p;

					matrix<double>* lambda_p = stochasticUpdateLambda();
					for(int l=0; l<K; l++)
						for(int m=0; m<K; m++)
                            (*lambda)(l,m) = ((1-stochastic_step_size)*(*lambda)(l,m) + stochastic_step_size*(*lambda_p)(l,m));
					delete lambda_p;

//					ll = getHeldoutLogLikelihood();
//					cout<<"held-ll "<<ll<<endl;//<<"\t ll ";

					//				cout<<"p,q "<<p<<" "<<q<<"||";
					//				printPhi(p,q);
				}
			}
//		ll = getHeldoutLogLikelihood();
//        cout<<"held-ll "<<ll<<endl;//<<"\t ll ";
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
//		for(int p=0; p<num_users; p++){
//			updateGamma(p);
			//			cout<<"update B now"<<endl;
			//			updateB();
			//			cout<<"updated both B and Gamma"<<endl;
//		}
		//		cout<<"Gamma\n";
		//		printMat(gamma, num_users, K);
//		updateNuFixedPoint();
//		updateLambda();
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
//		ll = getVariationalLogLikelihood();
		ll = getHeldoutLogLikelihood();
        cout<<"held-ll "<<ll<<endl;//<<"\t ll ";
//		cout<<ll<<endl;
		//       pi = gamma./repmat(sum(gamma,2),1,K)
	}
	return ll;
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
								&& heldUserAdjlist->at(user_thread)->count(userid_q)>0) )
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
//					stochasticUpdateGamma(p,q);
//					stochasticUpdateNuFixedPoint();
//					stochasticUpdateLambda();

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

void MMSBpoisson::initializeChiMats(){
	for(int k=0; k<K; ++k)
		for(int v=0; v<vocab_size; ++v)
			(*chi_kv_sum)(k,v)=0;
		
}

void MMSBpoisson::initializeChiPhiStatsOnce(){
	perUserThreadDelta = new std::unordered_map< std::pair<int,int>, int, class_hash<pair<int,int>>> ();
	perUserThreadChiStats4Phi = new std::unordered_map< std::pair<int,int>, std::vector<double>*, class_hash<pair<int,int>>> ();
	perUserThreadPhiStats4Chi = new std::unordered_map< std::pair<int,int>, std::vector<double>*, class_hash<pair<int,int>>> ();
	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>::iterator it1= userAdjlist->begin(); it1!=userAdjlist->end(); ++it1){
		std::vector<double>* chiStats = new std::vector<double>(K);
		std::vector<double>* phiStats = new std::vector<double>(K);
		for(int k=0; k<K; k++){
			chiStats->at(k)=0;
			phiStats->at(k)=0;
		}
		int userDelta = it1->second->size();			// users neighborhood size in the thread
		perUserThreadDelta->insert({it1->first, userDelta});
		perUserThreadPhiStats4Chi->insert({it1->first, phiStats});
		perUserThreadChiStats4Phi->insert({it1->first, chiStats});
	}
}

matrix<double>* MMSBpoisson::multiThreadStochasticUpdateNuFixedPoint(){
	matrix<double>* nu_p = new matrix<double>(K,K);
	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
			(*nu_p)(g,h) = (*nu)(g,h);
		}
	}
	double multiplier_temp = stochasticSamplePairMultiplier - multiThreadGlobalNetworkSampleSize;
	for(int it=0; it<nuIter; it++){
		for(int g=0; g<K; g++){
			for(int h=0; h<K; h++){
				double phi_nu = 0;
				double invTriGamma = 1/utils->trigamma((*nu_p)(g,h));
				double trigamma_gh = utils->trigamma((*nu_p)(g,h));
//				phi_nu+=(multiplier_temp*(*phi_y_gh_sum)(g,h)*trigamma_gh - 
//						multiplier_temp*(*phi_gh_sum)(g,h)*(*lambda)(g,h));
				phi_nu+=(stochasticSamplePairMultiplier*(*phi_y_gh_sum)(g,h)*trigamma_gh - 
						stochasticSamplePairMultiplier*(*phi_gh_sum)(g,h)*(*lambda)(g,h))/multiThreadGlobalNetworkSampleSize;
				//newPhis

//				cout<<"updateNuFixedPoint \n";
//				cout<<"nu: "<<phi_nu<<" "<<(*phi_y_gh_sum)(g,h)<<" "<<(*phi_gh_sum)(g,h)<<" ";
				phi_nu+=(((*kappa)(g,h) - (*nu_p)(g,h))*trigamma_gh + (1 - ((*lambda)(g,h)/(*theta)(g,h)))); 
//				cout<<(((*kappa)(g,h) - (*nu)(g,h))*trigamma_gh + (1 - ((*lambda)(g,h)/(*theta)(g,h))))<<endl;
				double temp_gh =0;
				while(temp_gh<=0){	
					temp_gh = (*nu_p)(g,h) + stepSizeNu*phi_nu;
//					cout<<"temp_gh "<<temp_gh<<" stepSizeNu "<<stepSizeNu<<" phi_nu "<<phi_nu<<"\t";
					if(temp_gh<=0)
						stepSizeNu*=0.1;
				}
				if(temp_gh>0){
//					cout<<"negative update in NU\t";
					(*nu_p)(g,h) = temp_gh;
				}
			}
		}
	}
	return nu_p;
}	

matrix<double>* MMSBpoisson::stochasticUpdateNuFixedPoint(){
	matrix<double>* nu_p = new matrix<double>(K,K);
	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
			(*nu_p)(g,h) = (*nu)(g,h);
		}
	}
	for(int it=0; it<nuIter; it++){
		for(int g=0; g<K; g++){
			for(int h=0; h<K; h++){
				double phi_nu = 0;
				double invTriGamma = 1/utils->trigamma((*nu_p)(g,h));
				double trigamma_gh = utils->trigamma((*nu_p)(g,h));
				phi_nu+=(stochasticSamplePairMultiplier*(*phi_y_gh_sum)(g,h)*trigamma_gh - 
						stochasticSamplePairMultiplier*(*phi_gh_sum)(g,h)*(*lambda)(g,h));//newPhis
//				cout<<"updateNuFixedPoint \n";
//				cout<<"nu: "<<phi_nu<<" "<<(*phi_y_gh_sum)(g,h)<<" "<<(*phi_gh_sum)(g,h)<<" ";
				phi_nu+=(((*kappa)(g,h) - (*nu_p)(g,h))*trigamma_gh + (1 - ((*lambda)(g,h)/(*theta)(g,h)))); 
//				cout<<(((*kappa)(g,h) - (*nu)(g,h))*trigamma_gh + (1 - ((*lambda)(g,h)/(*theta)(g,h))))<<endl; 
				(*nu_p)(g,h) = (*nu_p)(g,h) + stepSizeNu*phi_nu;
			}
		}
	}
	return nu_p; 
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




matrix<double>* MMSBpoisson::multiThreadStochasticUpdateTau(){
	matrix<double>* tau_p = new matrix<double>(K,vocab_size);
	double multiplier_temp = stochasticSamplePostsMultiplier/(1.0*multiThreadGlobalPostsSampleSize);
	for(int k=0; k<K; k++){
		double sum_v=0;
		for(int v=0; v<vocab_size; v++)                           {
			(*tau_p)(k,v) = ((*eta)(v) + multiplier_temp*(*chi_kv_sum)(k,v));
			sum_v+=(*tau_p)(k,v);
			if((*tau_p)(k,v)<=DBL_MIN || std::isnan((*tau_p)(k,v)) || std::isinf((*tau_p)(k,v))){
				cout<<"In multiThreadStochasticUpdateTau "<<(*chi_kv_sum)(k,v)<<"; "<<multiThreadGlobalPostsSampleSize<<endl;
			}

		}
//		for(int v=0; v<vocab_size; v++){
//			(*tau_p)(k,v) = (*tau_p)(k,v)/sum_v;
//		}
	}
	return tau_p;
}


void MMSBpoisson::updateLambda(){
	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
			(*lambda)(g,h) = ((*phi_y_gh_sum)(g,h)+(*kappa)(g,h))/(((*phi_gh_sum)(g,h) + 1/(*theta)(g,h))*(*nu)(g,h));  //newPhis
		}
	}

}

matrix<double>* MMSBpoisson::multiThreadStochasticUpdateLambda(){
	matrix<double>* lambda_p = new matrix<double>(K,K);
	double multiplier_temp = stochasticSamplePairMultiplier - multiThreadGlobalNetworkSampleSize;
	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
			(*lambda_p)(g,h) = (stochasticSamplePairMultiplier*(*phi_y_gh_sum)(g,h)/multiThreadGlobalNetworkSampleSize 
					+(*kappa)(g,h))/((stochasticSamplePairMultiplier*(*phi_gh_sum)(g,h)/multiThreadGlobalNetworkSampleSize 
						+ 1/(*theta)(g,h))*(*nu)(g,h));  //newPhis
//			(*lambda_p)(g,h) = (multiplier_temp*(*phi_y_gh_sum)(g,h)
//					+(*kappa)(g,h))/((multiplier_temp*(*phi_gh_sum)(g,h) 
//						+ 1/(*theta)(g,h))*(*nu)(g,h));  //newPhis
		}
	}

    return lambda_p;
}

matrix<double>* MMSBpoisson::stochasticUpdateLambda(){
	matrix<double>* lambda_p = new matrix<double>(K,K);
	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
			(*lambda_p)(g,h) = (stochasticSamplePairMultiplier*(*phi_y_gh_sum)(g,h)+(*kappa)(g,h))/((stochasticSamplePairMultiplier*(*phi_gh_sum)(g,h) + 1/(*theta)(g,h))*(*nu)(g,h));  //newPhis
		}
	}

    return lambda_p;
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
	int nuThresh=10;
	cout<<"nonDiagHyperInit: "<<nonDiagHyperInit<<"; diagHyperInit: "<<diagHyperInit<<endl<<flush;
	for(int g=0; g<K; g++)
		for(int h=0; h<K; h++){
			(*nu)(g,h)= nonDiagHyperInit;//0.05;//1;//rand()%nuThresh + 2;// 1;
			if(g==h) (*nu)(g,h)= diagHyperInit;//1;//rand()%nuThresh + 2;// 1;
		}
}

void MMSBpoisson::initializeLambda(){
	int lambdaThresh = 10;
	for(int g=0; g<K; g++)
		for(int h=0; h<K; h++){
			(*lambda)(g,h) = nonDiagHyperInit;//0.5;//rand()%lambdaThresh + 3;//1;
			if(g==h) (*lambda)(g,h)=diagHyperInit;//1;
		}
}

void MMSBpoisson::initializeTheta(){
	int thetaThresh=10;
	for(int g=0; g<K; g++)
		for(int h=0; h<K; h++){
			(*theta)(g,h)=nonDiagHyperInit;//0.5;//rand()%thetaThresh + 1;
			if(g==h) (*theta)(g,h)=diagHyperInit;//1;   
		}
}

void MMSBpoisson::initializeKappa(){
	int kappaThresh=10;
	for(int g=0; g<K; g++)
		for(int h=0; h<K; h++){
			(*kappa)(g,h)=nonDiagHyperInit;//0.5;
			if(g==h) (*kappa)(g,h)=diagHyperInit;//1;//2;//rand()%kappaThresh + 1;
		}
}


void MMSBpoisson::multiThreadedStochasticVariationalUpdatesPhi(int p, int q, int Y_pq, int thread_id, int Y_qp, 
		int threadID, pair<int,int> user_thread_p, pair<int,int> user_thread_q, int real_delta_tp){ // we have to take pair
//	cout<<"Start of method multiThreadedStochasticVariationalUpdatesPhi, threadID "<<threadID<<endl;
	double digamma_p_sum = getDigamaValue(getMatrixRowSum(gamma,p,K));
	double digamma_q_sum = getDigamaValue(getMatrixRowSum(gamma,q,K));

	//TODO: update Phi4Chi stats	

	matrix<double>* phi_gh_pq = new matrix<double>(K,K);
	matrix<double>* phi_gh_qp = new matrix<double>(K,K);
	boost::numeric::ublas::vector<double>* phi_pg_update = new boost::numeric::ublas::vector<double>(K);
	boost::numeric::ublas::vector<double>* phi_qh_update = new boost::numeric::ublas::vector<double>(K);
	boost::numeric::ublas::vector<double>* phi_pg_update_q = new boost::numeric::ublas::vector<double>(K);
	boost::numeric::ublas::vector<double>* phi_qh_update_q = new boost::numeric::ublas::vector<double>(K);
//	cout<<digamma_q_sum<<endl;
//	cout<<digamma_p_sum<<endl;
	
	std::vector<int>* randomNonDiags = new std::vector<int>(K);
	std::vector<int>* randomNonDiags_q = new std::vector<int>(K);

	double phi_sum = 0;
	double phi_sum_q = 0;
	
	double chiStats_p =0, chiStats_q=0, delta_tp=0, delta_tq=0;
	std::vector<double>* chi_vec_p, *chi_vec_q;
	if(textFactorForNWTextBalance>0){
		if(perUserThreadDelta->count(user_thread_p)>0)
			delta_tp = perUserThreadDelta->at(user_thread_p);
		if(perUserThreadDelta->count(user_thread_q)>0)
			delta_tq = perUserThreadDelta->at(user_thread_q);
		if(delta_tp>0){
			double log_epsilon_delta = chi_epsilon/(1.0*delta_tp);
//			if(log_epsilon_delta<1)
//				log_epsilon_delta = const_log_epsilon_delta;				//TODO: make it a constant
//			chiStats_p = (-1.0*log(log_epsilon_delta)*(1.0/delta_tp) + log(1 + log_epsilon_delta)*(1.0/delta_tp));
//			chiStats_p = (-1.0*log(log_epsilon_delta) + log(1 + log_epsilon_delta));
			chi_vec_p = perUserThreadChiStats4Phi->at(user_thread_p);
		}
		if(delta_tq>0){
//			double log_epsilon_delta = chi_epsilon/(1.0*delta_tq);
//			if(log_epsilon_delta<1)
//				log_epsilon_delta = const_log_epsilon_delta;				//TODO: make it a constant
//			chiStats_q = (-1.0*log(log_epsilon_delta)*(1.0/delta_tq) + log(1 + log_epsilon_delta)*(1.0/delta_tq));
//			chiStats_q = (-1.0*log(log_epsilon_delta) + log(1 + log_epsilon_delta));
			chi_vec_q = perUserThreadChiStats4Phi->at(user_thread_q);
		}
	}

	if(real_delta_tp!=delta_tp){
		cout<<endl<<"REAL_DELTA_TP!=DELTA_TP: "<<real_delta_tp<<"!="<<delta_tp<<endl;
		exit(0);
	}

	for(int g=0;g<K;g++){
		double chi_p=0, chi_q=0;
		if(delta_tp>0)
			chi_p = textFactorForNWTextBalance*chi_vec_p->at(g);
//			chi_p = textFactorForNWTextBalance*chiStats_p*chi_vec_p->at(g);
		if(delta_tq>0)
			chi_q = textFactorForNWTextBalance*chi_vec_q->at(g);
//			chi_q = textFactorForNWTextBalance*chiStats_q*chi_vec_q->at(g);

		int h=g;
		int rand_indx = rand()%(K-1);
//TODO:	need to change the above code line; the rand() starts form 1
		if(rand_indx==g)
			rand_indx++;
//		cout<<"\tasdf: "<<rand_indx<<"\t"<<endl;
		randomNonDiags->at(g) = rand_indx;
			(*phi_gh_pq)(g,rand_indx) = exp(dataFunctionPhiUpdates(g,rand_indx,Y_pq) 
				+ (getDigamaValue((*gamma)(p,g)) - digamma_p_sum)
				+ (getDigamaValue((*gamma)(q,rand_indx)) - digamma_q_sum) + chi_p);
			if((*phi_gh_pq)(g,rand_indx)>= DBL_MAX)
				(*phi_gh_pq)(g,rand_indx) = DBL_MAX/(K*K + K) ;
			phi_sum+=(*phi_gh_pq)(g,rand_indx);

//		for(int h=0;h<K;h++){
			(*phi_gh_pq)(g,h) = exp(dataFunctionPhiUpdates(g,h,Y_pq) 
				+ (getDigamaValue((*gamma)(p,g)) - digamma_p_sum)
				+ (getDigamaValue((*gamma)(q,h)) - digamma_q_sum) + chi_p);
			if((*phi_gh_pq)(g,h)>= DBL_MAX)
				(*phi_gh_pq)(g,h) = DBL_MAX/(K*K + K) ;
			phi_sum+=(*phi_gh_pq)(g,h);

			(*phi_gh_qp)(g,h) = exp(dataFunctionPhiUpdates(g,h,Y_qp) 
				+ (getDigamaValue((*gamma)(q,g)) - digamma_q_sum)
				+ (getDigamaValue((*gamma)(p,h)) - digamma_p_sum) + chi_q);

			if((*phi_gh_qp)(g,h)>= DBL_MAX)
				(*phi_gh_qp)(g,h) = DBL_MAX/(K*K + K) ;
			phi_sum_q += (*phi_gh_qp)(g,h);

			if(std::isnan((*phi_gh_pq)(g,h)) || (*phi_gh_pq)(g,h)<0 || (*phi_gh_qp)(g,h) <0 || std::isnan((*phi_gh_qp)(g,h))
					|| std::isinf((*phi_gh_pq)(g,h)) ||std::isinf((*phi_gh_qp)(g,h))){
				cout<<"In variational Phi updates "<<p<<" "<<q<<" "<<Y_pq<<" "<<thread_id<<" "<<(*phi_gh_pq)(g,h)<<" "<<(*phi_gh_qp)(g,h)
					<<"; "<<chi_p<<"; "<<chi_q<<"; "<<delta_tp<<"; "<<delta_tq<<"; "<<chiStats_p<<"; "<<chiStats_q<<"; "<<chi_vec_p->at(g)<<endl;
				exit(0);
			}
//		}
		rand_indx = rand()%(K-1);
//TODO:	need to change the above code line; the rand() starts form 1
		if(rand_indx==g)
			rand_indx++;
		randomNonDiags_q->at(g) = rand_indx;

		//		cout<<"\tasdf: "<<rand_indx<<"\t"<<endl;
		
			(*phi_gh_qp)(g,rand_indx) = exp(dataFunctionPhiUpdates(g,rand_indx,Y_qp) 
				+ (getDigamaValue((*gamma)(q,g)) - digamma_q_sum)
				+ (getDigamaValue((*gamma)(p,rand_indx)) - digamma_p_sum) + chi_q);
			if((*phi_gh_qp)(g,rand_indx)>= DBL_MAX)
				(*phi_gh_qp)(g,rand_indx) = DBL_MAX/(K*K + K) ;
			phi_sum_q += (*phi_gh_qp)(g,rand_indx);

//			if(std::isnan((*phi_gh_pq)(g,h))){
//				cout<<p<<" "<<q<<" "<<Y_pq<<" "<<thread_id<<" "<<(*phi_gh_pq)(g,h)<<endl;
//			}

		(*phi_qh_update)(g) = 0;
		(*phi_pg_update)(g) = 0;
		(*phi_qh_update_q)(g) = 0;
		(*phi_pg_update_q)(g) = 0;
	}

//	cout<<"variationalUpdatesPhi"<<endl;
	
	double temp_phi_gh = 0;
	double temp_phi_gh_q = 0;

	for(int g=0;g<K;g++){
		int h=g;
			if(std::isnan((*phi_pg_update)(g))){
				cout<<"First nan in phi_pg_update"<<(*phi_pg_update)(g)<<"; "<<g<<"; "<<(*phi_gh_pq)(g,h)<<"; "<<phi_sum<<endl;
				exit(0);
			}
//		for(int h=0;h<K;h++){
			temp_phi_gh=(*phi_gh_pq)(g,h);
			temp_phi_gh_q = (*phi_gh_qp)(g,h);

			int rand_indx = randomNonDiags->at(g);
			if(phi_sum<=DBL_MIN){
//				cout<<"In DBL_MIN "<<DBL_MIN<<endl;
				(*phi_gh_pq)(g,h) = 1.0/(2.0*K);//(K*K);//(2*K);	// NOTE: not K*K
				(*phi_gh_pq)(g,rand_indx) = 1.0/(2*K);	// NOTE: not K*K
				phi_sum=1;
			}else{
  			(*phi_gh_pq)(g,h) = ((*phi_gh_pq)(g,h))/phi_sum ;
  			(*phi_gh_pq)(g,rand_indx) = ((*phi_gh_pq)(g,rand_indx))/phi_sum ;
			}

			int rand_indx_q = randomNonDiags_q->at(g);
			if(phi_sum_q<=DBL_MIN){
//				cout<<"In DBL_MIN "<<DBL_MIN<<endl;
				(*phi_gh_qp)(g,h) = 1.0/(2.0*K);//(K*K);//(2*K);
				(*phi_gh_qp)(g,rand_indx_q) = 1.0/(2*K);
				phi_sum_q=1;
			}else{
				(*phi_gh_qp)(g,h) = ((*phi_gh_qp)(g,h))/phi_sum_q ;
				(*phi_gh_qp)(g,rand_indx_q) = ((*phi_gh_qp)(g,rand_indx_q))/phi_sum_q ;
			}

			(*phi_gh_sum_thread_list->at(threadID))(g,h) += ((*phi_gh_pq)(g,h) + (*phi_gh_qp)(g,h));
			(*phi_y_gh_sum_thread_list->at(threadID))(g,h) += ((*phi_gh_pq)(g,h)*Y_pq + (*phi_gh_qp)(g,h)*Y_qp);
			(*phi_gh_sum_thread_list->at(threadID))(g,rand_indx) += ((*phi_gh_pq)(g,rand_indx)); //dont add rand_indx_q
			(*phi_y_gh_sum_thread_list->at(threadID))(g,rand_indx) += ((*phi_gh_pq)(g,rand_indx)*Y_pq);
			(*phi_gh_sum_thread_list->at(threadID))(g,rand_indx_q) += ((*phi_gh_qp)(g,rand_indx_q));
			(*phi_y_gh_sum_thread_list->at(threadID))(g,rand_indx_q) += ((*phi_gh_qp)(g,rand_indx_q)*Y_qp);

//			if(std::isnan((*phi_gh_pq)(g,h))||(*phi_gh_pq)(g,h)<0)
			if(std::isnan(phi_sum)||phi_sum<=DBL_MIN||std::isnan(phi_sum_q)||phi_sum_q<=DBL_MIN){
				cout<<"\nVariationalUpdate "<<p<<"i; "<<q<<"; "<<Y_pq<<"; "<<thread_id<<"; "<<(*phi_gh_pq)(g,h)<<"; "<<temp_phi_gh<<"; "<<dataFunctionPhiUpdates(g,h,Y_pq)<<"; "<< exp(dataFunctionPhiUpdates(g,h,Y_pq))<<"; "<<phi_sum<<"; DBL_MIN "<<DBL_MIN<<endl;
				exit(0);
			}
			(*phi_qh_update)(h) += ((*phi_gh_pq)(g,h));
			(*phi_qh_update_q)(h) += (*phi_gh_qp)(g,h);
			(*phi_qh_update)(rand_indx) += ((*phi_gh_pq)(g,rand_indx));
			(*phi_qh_update_q)(rand_indx_q) += (*phi_gh_qp)(g,rand_indx_q);

			(*phi_pg_update)(g) += (*phi_gh_pq)(g,h);
			(*phi_pg_update_q)(g) += (*phi_gh_qp)(g,h);
			if(std::isnan((*phi_pg_update)(g))){
				cout<<" nan in phi_pg_update"<<(*phi_pg_update)(g)<<"; "<<(*phi_gh_pq)(g,h)<<"; "<<temp_phi_gh<<endl;
				exit(0);
			}
			(*phi_pg_update)(rand_indx) += (*phi_gh_pq)(g,rand_indx);
			(*phi_pg_update_q)(rand_indx_q) += (*phi_gh_qp)(g,rand_indx_q);
//		}
	}

//	for(int h=0; h<K; h++){
//		for(int g=0; g<K; g++){
//			(*phi_qh_update)(h) += ((*phi_gh_pq)(g,h));
//			(*phi_qh_update_q)(h) += (*phi_gh_qp)(g,h);
//		}
//	}
//
//	for(int g=0; g<K; g++){
//		for(int h=0; h<K; h++){
//			(*phi_pg_update)(g) += (*phi_gh_pq)(g,h);
//			(*phi_pg_update_q)(g) += (*phi_gh_qp)(g,h);
//		}
//	}

	for(int k=0; k<K; k++){
		(*phi_pg_sum_thread_list->at(threadID))(p,k) += ((*phi_pg_update)(k));

        // TODO initialize the vectors and set them to 0
		if(Y_pq>0 && real_delta_tp>0 && delta_tp>0){

//TODO: Here we divide by delta_tp whereas in chi updates we divide by delta_tp
			perUserThreadPhiStats4Chi_thread_list->at(threadID)->at(user_thread_p)->at(k) += (((*phi_pg_update)(k))/delta_tp);
			if(std::isnan(perUserThreadPhiStats4Chi_thread_list->at(threadID)->at(user_thread_p)->at(k)) || perUserThreadPhiStats4Chi_thread_list->at(threadID)->at(user_thread_p)->at(k) <0){
				cout<<"invalid value in the stats update "<<perUserThreadPhiStats4Chi_thread_list->at(threadID)->at(user_thread_p)->at(k)<<"; "<<(((*phi_pg_update)(k))/delta_tp)<<"; "<<delta_tp<<"; "<<real_delta_tp<<"; "<<((*phi_pg_update)(k))<<endl;
				exit(0);
			}
		}

		(*phi_qh_sum_thread_list->at(threadID))(q,k) += ((*phi_qh_update)(k));
		(*phi_pg_sum_thread_list->at(threadID))(q,k) += ((*phi_pg_update_q)(k));
		(*phi_qh_sum_thread_list->at(threadID))(p,k) += ((*phi_qh_update_q)(k));
	}

	delete phi_gh_pq;
	delete phi_gh_qp;
	delete phi_pg_update;
	delete phi_qh_update;
	delete phi_pg_update_q;
	delete phi_qh_update_q;
	delete randomNonDiags;
	delete randomNonDiags_q;
//	cout<<"End of method multiThreadedStochasticVariationalUpdatesPhi, threadID "<<threadID<<endl;
}

void MMSBpoisson::stochasticVariationalUpdatesPhi(int p, int q, int Y_pq, int thread_id, int Y_qp){
	//N = size(Y,1);
	//K = size(alpha,2);
//	boost::numeric::ublas::vector<double>* deriv_phi_p = new boost::numeric::ublas::vector<double>(K);
//	boost::numeric::ublas::vector<double>* deriv_phi_q = new boost::numeric::ublas::vector<double>(K);
//	int Y_pq = (*inputMat)(p,q);
	double digamma_p_sum = getDigamaValue(getMatrixRowSum(gamma,p,K));
	double digamma_q_sum = getDigamaValue(getMatrixRowSum(gamma,q,K));
	

	matrix<double>* phi_gh_pq = new matrix<double>(K,K);
	matrix<double>* phi_gh_qp = new matrix<double>(K,K);
	boost::numeric::ublas::vector<double>* phi_pg_update = new boost::numeric::ublas::vector<double>(K);
	boost::numeric::ublas::vector<double>* phi_qh_update = new boost::numeric::ublas::vector<double>(K);
	boost::numeric::ublas::vector<double>* phi_pg_update_q = new boost::numeric::ublas::vector<double>(K);
	boost::numeric::ublas::vector<double>* phi_qh_update_q = new boost::numeric::ublas::vector<double>(K);
//	cout<<digamma_q_sum<<endl;
//	cout<<digamma_p_sum<<endl;

	double phi_sum = 0;
	double phi_sum_q = 0;
	for(int g=0;g<K;g++){
		for(int h=0;h<K;h++){
			(*phi_gh_pq)(g,h) = exp(dataFunctionPhiUpdates(g,h,Y_pq) 
				+ (getDigamaValue((*gamma)(p,g)) - digamma_p_sum)
				+ (getDigamaValue((*gamma)(q,h)) - digamma_q_sum));
			phi_sum+=(*phi_gh_pq)(g,h);

			(*phi_gh_qp)(g,h) = exp(dataFunctionPhiUpdates(g,h,Y_qp) 
				+ (getDigamaValue((*gamma)(q,g)) - digamma_q_sum)
				+ (getDigamaValue((*gamma)(p,h)) - digamma_p_sum));
			phi_sum_q += (*phi_gh_qp)(g,h);
//			if(std::isnan((*phi_gh_pq)(g,h))){
//				cout<<p<<" "<<q<<" "<<Y_pq<<" "<<thread_id<<" "<<(*phi_gh_pq)(g,h)<<endl;
//			}
		}
		(*phi_qh_update)(g) = 0;
		(*phi_pg_update)(g) = 0;
		(*phi_qh_update_q)(g) = 0;
		(*phi_pg_update_q)(g) = 0;
	}

//	cout<<"variationalUpdatesPhi"<<endl;
	
	double temp_phi_gh = 0;
	double temp_phi_gh_q = 0;

	for(int g=0;g<K;g++){
		for(int h=0;h<K;h++){
			temp_phi_gh=(*phi_gh_pq)(g,h);
			temp_phi_gh_q = (*phi_gh_qp)(g,h);

			if(phi_sum<DBL_MIN){
//				cout<<"In DBL_MIN "<<DBL_MIN<<endl;
				(*phi_gh_pq)(g,h) = 1.0/(K*K);
			}else
				(*phi_gh_pq)(g,h) = ((*phi_gh_pq)(g,h))/phi_sum ;

			if(phi_sum_q<DBL_MIN){
//				cout<<"In DBL_MIN "<<DBL_MIN<<endl;
				(*phi_gh_qp)(g,h) = 1.0/(K*K);
			}else
				(*phi_gh_qp)(g,h) = ((*phi_gh_qp)(g,h))/phi_sum_q ;

			(*phi_gh_sum)(g,h) = ((*phi_gh_pq)(g,h) + (*phi_gh_qp)(g,h));
			(*phi_y_gh_sum)(g,h) = ((*phi_gh_pq)(g,h)*Y_pq + (*phi_gh_qp)(g,h)*Y_qp);//((*inputMat)(p,q)));

			if(std::isnan((*phi_gh_pq)(g,h))||(*phi_gh_pq)(g,h)<0)
				cout<<"VariationalUpdate "<<p<<" "<<q<<" "<<Y_pq<<" "<<thread_id<<" "<<(*phi_gh_pq)(g,h)<<" "<<temp_phi_gh<<" "<<dataFunctionPhiUpdates(g,h,Y_pq)<<" "<< exp(dataFunctionPhiUpdates(g,h,Y_pq))<<phi_sum<<" DBL_MIN "<<DBL_MIN<<endl;
		}
	}
	for(int h=0; h<K; h++){
		for(int g=0; g<K; g++){
			(*phi_qh_update)(h) += ((*phi_gh_pq)(g,h));
			(*phi_qh_update_q)(h) += (*phi_gh_qp)(g,h);
		}
	}

	for(int g=0; g<K; g++){
		for(int h=0; h<K; h++){
			(*phi_pg_update)(g) += (*phi_gh_pq)(g,h);
			(*phi_pg_update_q)(g) += (*phi_gh_qp)(g,h);
		}
	}

	for(int k=0; k<K; k++){
		(*phi_pg_sum)(p,k) = ((*phi_pg_update)(k));
		(*phi_qh_sum)(q,k) = ((*phi_qh_update)(k));
		(*phi_pg_sum)(q,k) += ((*phi_pg_update_q)(k));
		(*phi_qh_sum)(p,k) += ((*phi_qh_update_q)(k));
	}
	delete phi_gh_pq;
	delete phi_gh_qp;
	delete phi_pg_update;
	delete phi_qh_update;
	delete phi_pg_update_q;
	delete phi_qh_update_q;
}

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
			if(phi_sum<DBL_MIN)
				(*phi_gh_pq)(g,h) = 1.0/(K*K*1.0);
			else
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




void MMSBpoisson::initializeUserIndex(unordered_map<int,int>* userList){  
	for(std::unordered_map<int,int>::iterator it=userList->begin(); it!=userList->end(); it++){
		userIndexMap->insert({it->second, it->first});
	}	
}

void MMSBpoisson::initializeGamma(){
//	cout<<"In InitializeGamma\n";
	for (int p = 0; p < num_users; ++p) {
//		cout<<p<<" ";
		int rand_indx = rand()%K;
		for (int k = 0; k < K; ++k) {
//			(*gamma)(p,k)=(*alpha)(k)+abs((getUniformRandom()-0.5)*0.1);
			(*gamma)(p,k) = (getUniformRandom()/(K*K*1.0));
//			cout<<(*alpha)(k)<<" "<<(*gamma)(p,k)<<" ";

		}
//		(*gamma)(p,rand_indx) += (*gamma)(p,rand_indx) + K;
//		cout<<endl;
	}
	if(strcmp(seedIndexFileName,"null"))
		utils->intializePiFromIndexFile(gamma, seedIndexFileName, userList);
    matrix<double>* init_pis = getPis();
	char temp_str[] = "init_graclus_Pi";
	cout<<temp_str<<endl;
	printPiToFile(init_pis, num_users, K, temp_str, userIndexMap);
	delete init_pis;
}

void MMSBpoisson::initializeTau(){
//	cout<<"In InitializeGamma\n";
	for (int k = 0; k < K; ++k) {
//		cout<<p<<" ";
//		int rand_indx = rand()%K;
		for (int v = 0; v < vocab_size; ++v) {
			(*tau)(k,v)=abs((*eta)(v)+(getUniformRandom()-0.5)*0.1);
//			cout<<(*alpha)(k)<<" "<<(*gamma)(p,k)<<" ";

		}
//		(*gamma)(p,rand_indx) += (*gamma)(p,rand_indx) + 5;
//		cout<<endl;
	}
}

matrix<double>* MMSBpoisson::stochasticUpdateGamma(int p, int q){
//	boost::numeric::ublas::vector<double>* gamma_p = new boost::numeric::ublas::vector<double>(K);
	matrix<double>* gamma_pq = new matrix<double>(2,K);
	for(int k=0; k<K; k++){
		(*gamma_pq)(0,k) = (*alpha)(k);
		(*gamma_pq)(0,k) += (stochasticSampleNodeMultiplier*(*phi_pg_sum)(p,k) + 
				stochasticSampleNodeMultiplier*(*phi_qh_sum)(p,k));	// new Phis
		
		(*gamma_pq)(1,k) = (*alpha)(k);
		(*gamma_pq)(1,k) += (stochasticSampleNodeMultiplier*(*phi_pg_sum)(q,k) + 
				stochasticSampleNodeMultiplier*(*phi_qh_sum)(q,k));	// new Phis
//		cout<<"gamma_pk "<<p<<","<<k<<","<<gamma_pk<<";";
	}
//	cout<<endl;
	return gamma_pq;
}


boost::numeric::ublas::vector<double>* MMSBpoisson::multiThreadStochasticUpdateGamma(int p){
	boost::numeric::ublas::vector<double>* gamma_p = new boost::numeric::ublas::vector<double>(K);
//	matrix<double>* gamma_pq = new matrix<double>(2,K);
	double multiplier_temp = stochasticSampleNodeMultiplier - multiThreadGlobalNetworkSampleSize;
	for(int k=0; k<K; k++){
//		cout<<"alpha_k "<<(*alpha)(k)<<" ";
		(*gamma_p)(k) = (*alpha)(k);
		(*gamma_p)(k) += (stochasticSampleNodeMultiplier*(*phi_pg_sum)(p,k) + 
				stochasticSampleNodeMultiplier*(*phi_qh_sum)(p,k))/multiThreadGlobalNetworkSampleSize;	// new Phis
//		(*gamma_p)(k) += (multiplier_temp*(*phi_pg_sum)(p,k) + 
//				multiplier_temp*(*phi_qh_sum)(p,k));	// new Phis

//		if((*phi_pg_sum)(p,k) <=0 ||(*phi_qh_sum)(p,k) <=0 )
//			cout<<p<<" "<<k<<" "<<" "<<(*phi_pg_sum)(p,k)<<" "<<(*phi_qh_sum)(p,k)<< " "<<(*gamma_p)(k)<<
//				" "<<stochasticSampleNodeMultiplier<<" "<<multiThreadGlobalNetworkSampleSize<<endl ;
		
//		(*gamma_pq)(1,k) = (*alpha)(k);
//		(*gamma_pq)(1,k) += (stochasticSampleNodeMultiplier*(*phi_pg_sum)(q,k) + 
//				stochasticSampleNodeMultiplier*(*phi_qh_sum)(q,k));	// new Phis
//		cout<<"gamma_pk "<<p<<","<<k<<","<<gamma_pk<<";";
	}
//	cout<<endl;
	return gamma_p;
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
	std::unordered_set<int>* vocabList = new std::unordered_set<int>();
	std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<pair<int,int>>>* userAdjlist = 
		new std::unordered_map<std::pair<int,int>,std::unordered_map<int,int>*, class_hash<pair<int,int>>>();
	std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost = 
		new std::unordered_map<std::pair<int,int>,std::vector<int>*, class_hash<pair<int,int>>>();

//	username_mention_graph.txt

	utilsClass->readThreadStructureFile(argv[1], userList, threadList, vocabList, userAdjlist, userThreadPost);

	std::unordered_map< std::pair<int,int>, std::unordered_map<int,std::pair<int,int>>*, class_hash<pair<int,int>>>* heldUserAdjlist = 
		new std::unordered_map<std::pair<int,int>,std::unordered_map<int,std::pair<int,int>>*, class_hash<pair<int,int>>>();
	std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<pair<int,int>>>* heldUserAdjlist_held = 
		new std::unordered_map<std::pair<int,int>,std::unordered_map<int,int>*, class_hash<pair<int,int>>>();

	std::unordered_map<int, int>* userIndexMap = initializeUserIndex(userList);
	std::unordered_map<int, std::unordered_set<int>*>* perThreadUserSet = getPerThreadUserSet(userAdjlist);

	std::unordered_map<int, std::string>* vocabMap = new std::unordered_map<int, std::string>();
	char* vocabFile = argv[19];                     
	utilsClass->readVocabMap(vocabMap, vocabFile);

//	std::pair<int,int> numHeldAndTotalEdges = utilsClass->getTheHeldoutSet(userAdjlist, heldUserAdjlist, 0.05, perThreadUserSet, userList->size(), userIndexMap, heldUserAdjlist_held, argv[13]);
	
//	heldUserAdjlist_held->clear();
//	heldUserAdjlist->clear();
	std::pair<int,int> numHeldAndTestEdges = utilsClass->readHeldoutAndTest(heldUserAdjlist, heldUserAdjlist_held, argv[13]);

	delete userIndexMap;

	int numHeldoutEdges = numHeldAndTestEdges.first;
	int numTotalLinks = 28800;//numHeldAndTotalEdges.second;	//this should be alculated in readThreadStructureFile method
	int numTestEdges = numHeldAndTestEdges.second;
//	cout<< "numTestEdges "<< numTestEdges<<endl;

//	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int,std::pair<int,int>>*, class_hash<pair<int,int>>>::iterator it1=heldUserAdjlist->begin(); it1!=heldUserAdjlist->end(); ++it1){
//		int U1 = it1->first.first;
//		int threadId = it1->first.second;
//		for(std::unordered_map<int,std::pair<int,int>>::iterator it2 = it1->second->begin(); it2!=it1->second->end(); ++it2){
//			int U2 = it2->first;
//			int count = it2->second.second;
//			int testOrHeldEdges = it2->second.first;
//			cout << testOrHeldEdges <<" "<<U1<<" "<<U2<<" "<<threadId<<" "<<count<<" "<<"NOPOST"<<endl;
//		}
//	}

//	testDataStructures(userList,threadList, userAdjlist,userThreadPost);

//	cout<<endl<<i<<" "<<INT_MAX<<endl;

	cout<<"Before MMSB constructor"<<endl;

	MMSBpoisson* mmsb = new MMSBpoisson(utilsClass);

//	cout<<"after MMSB constructor call"<<endl;
//	mmsb->getParameters(matFile, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
	double stepSizeNu = atof(argv[6]);
	int vocabSize = atoi(argv[12]);
	double textFactorForNWTextBalance = atof(argv[14]);		
	int zeroEdges = atoi(argv[15]);                         
	double diagHyperInit = atof(argv[16]);
	double nonDiagHyperInit = atof(argv[17]);                  
	char* seedIndexFileName = argv[18];                     
	double alpha = atof(argv[20]);                   
	if(argc>21){
		bool fasterFlag = (atoi(argv[21])==1);                 
		mmsb->setFasterFlag(fasterFlag);
	}
	if(argc>22){                                               //LAST INDEX USED
		int constantThreads=atoi(argv[22]);
		mmsb->setConstantThreads(constantThreads);
	}
	if(argc>23){                                               //LAST INDEX USED
		double nwFactorForText =atof(argv[23]);
		mmsb->setNwFactorForText(nwFactorForText);
	}
	if(argc>24){                                               //LAST INDEX USED
		bool printFullLDATopics =(atoi(argv[24])==1);
		mmsb->setPrintFullLDATopics(printFullLDATopics);
	}
	mmsb->setZeroEdges(zeroEdges);
	mmsb->setHyperInits(diagHyperInit, nonDiagHyperInit);
	mmsb->setSeedIndexFileName(seedIndexFileName);
	mmsb->setVocabMap(vocabMap);
	mmsb->initialize(K, userList, threadList, vocabList, userAdjlist, heldUserAdjlist, 
			userThreadPost, stepSizeNu, numHeldoutEdges, atof(argv[7]), atof(argv[8]), atoi(argv[10]), vocabSize, heldUserAdjlist_held);
	mmsb->initializeAlpha(alpha);
//	mmsb->getParameters(atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
	mmsb->getParametersInParallel(atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[9]), argv[11], perThreadUserSet, numTotalLinks, textFactorForNWTextBalance);

	delete userList;
	delete userAdjlist;
	delete threadList;
	delete userThreadPost;

	delete perThreadUserSet;

//void MMSBpoisson::initialize(int K, std::unordered_map<int,int>* userList, std::unordered_set<int>* threadList,  
//	std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* userAdjlist,
//	std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* heldUserAdjlist,
//	std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost,
//	double stepSizeNu, int numHeldoutEdges, double stochastic_step_kappa, double samplingThreshold, 
//	int numParallelThreads){        


//void MMSBpoisson::getParametersInParallel(int iter_threshold, int inner_iter, int nu_iter, 
//		int stochastic_tau, char* outputFile){ // TODO: change phi coz it is K*K*N*N 


//	cout<<mmsb->getLnGamma(atoi(argv[1]));

//	cout<<mmsb->getDigamaValue(20)<<endl;
//	cout<<mmsb->getDigamaValue(30)<<endl;
//	cout<<mmsb->getDigamaValue(10)<<endl;
//	cout<<mmsb->getDigamaValue(1)<<endl;
//	cout<<mmsb->getDigamaValue(0)<<endl;
//	cout<<mmsb->getDigamaValue(101)<<endl;
	return 0;



}
