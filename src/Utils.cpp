/*
 * Utils.cpp
 *
 *  Created on: Apr 26, 2013
 *      Author: abhimank
 */

#include <boost/numeric/ublas/matrix.hpp>
#include "csv_parser.hpp"
#include "Utils.h"
#include <iostream>
#include <boost/tr1/functional.hpp>
#include <boost/algorithm/string.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <boost/random/uniform_01.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>
//#include <boost/algorithm/string.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace boost::numeric::ublas;
matrix<int> *Utils::readCsvToMat(char* filename, int numRows, int numColumns) {

	//		const char * filename = "monkLK.csv";
	const char field_terminator = ',';
	const char line_terminator = '\n';
	const char enclosure_char = '"';

	matrix<int>* matFile = new matrix<int>(numRows, numColumns);

	csv_parser file_parser;
	file_parser.set_skip_lines(0);
	file_parser.init(filename);

	file_parser.set_enclosed_char(enclosure_char, ENCLOSURE_OPTIONAL);
	file_parser.set_field_term_char(field_terminator);
	file_parser.set_line_term_char(line_terminator);

	unsigned int row_curr = 0;

	for(int i=0; i<numRows; i++){	//file_parser.has_more_rows()){
		csv_row row = file_parser.get_row();
		for (int j = 0; j < numColumns; j++){//row.size(); ++i) {
			matFile->operator ()(i,j)=atoi(row[j].c_str());
			//				cout << atoi(row[j].c_str())<<" ";
		}
		//			cout<<endl;
		//				cout<<matFile(1,1);
		}
		return matFile;
	}

//template <class T>
//class class_hash{
//	const size_t bucket_size = 10; //
//	const size_t min_buckets = (1<<10);
//	
//	size_t operator()(const T& key_value) const{
//		std::tr1::hash<std::string> hash_fn;
//		string keyVal = itoa(key_value->first) + itoa(key_value->second);
//		return hash_fn(keyVal);
//	}
//
//    bool operator()(const T& left, const T& right) const{
//		return (((int)left->first < (int)right->first)||
//		 (((int)left->first < (int)right->first)&&((int)left->first < (int)right->first)));
//	}
//};

void Utils::readVocabMap(std::unordered_map<int, string>* vocabMap, char* fileName){
	char s[20000];
	FILE* filePointer = fopen(fileName,"r");
	int index=1;
	while(fgets(s, 20000, filePointer)!=NULL){
		//boost::algorithm::trim(s);
		vocabMap->insert({index, s});
		index++;
	}
}

void Utils::getSeedClusters(char* fileName, std::unordered_map<int,std::vector<int>*>* seedSetMap, 
		std::unordered_set<int>* uniqueSeedSet){
	char s[20000];
	FILE* seedPointer = fopen(fileName,"r");
	int index=0;
	while(fgets(s, 20000, seedPointer)!=NULL){
		std::vector<int>* seedIndex = new std::vector<int>();
		std::vector<std::string> seedIds; 
		boost::split(seedIds, s, boost::is_any_of("\t "));
		addWords(seedIndex, &seedIds, uniqueSeedSet);
		seedSetMap->insert({index, seedIndex});
		index++;
	}
}


void Utils::getDataStats(
std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*,class_hash<pair<int,int>>>* completeUserAdjlist, 
std::unordered_map<int,std::unordered_set<int>*>* perThreadUserSet, int num_users, char* filename){

	int totalPosts = 0;
	int totalEdgeCounts = 0;
//	int attemptThreshold = -1;//10;
	int histLimit = 10;
	std::vector<double>* edgeHistogram = new std::vector<double>(histLimit+1); // edge weights 1, 2, 3, 4, 5, 6, 7 8 9 10 above

	for(int wt=0; wt<histLimit+1; wt++){
		edgeHistogram->at(wt)=0;
	}
	ofstream outfile(filename);
	cout<<"after inititalization\n";

	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<pair<int,int>>>::iterator it1=completeUserAdjlist->begin(); it1!=completeUserAdjlist->end(); ++it1){
//		if(it1->second->size()<=1)
		for(std::unordered_map<int,int>::iterator it2 = it1->second->begin(); it2!=it1->second->end(); ++it2){
			totalPosts += it2->second;		
			totalEdgeCounts++;
			if(it2->second<=10)
				edgeHistogram->at(it2->second-1) +=1;
			else
				edgeHistogram->at(10) +=1;
		}
	}
	outfile<<"TotalPosts: "<<totalPosts<<endl;
	outfile<<"TotalEdges: "<<totalEdgeCounts<<endl;
	outfile<<"numUsers: "<<num_users<<endl;
	outfile<<"numThreads: "<<perThreadUserSet->size()<<endl;
	outfile<<"=======EdgeHistograms======\n";
	for(int wt=0; wt<histLimit+1; wt++){
		outfile<<wt+1<<","<<edgeHistogram->at(wt)<<endl;
	}
	outfile.close();
}

void Utils::getNodeDegree(char* piFile,char* degreeFile,std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*,class_hash<pair<int,int>>>* completeUserAdjlist){
	FILE* piFileDescriptor = fopen(piFile, "r");
	int user;
	char s[20000];
	std::unordered_map<int,int>* nodeDegreeMap = new unordered_map<int, int>();
	long total_counts = 0;
	ofstream outfile(degreeFile);

	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<std::pair<int,int>>>::iterator it1=completeUserAdjlist->begin(); it1!=completeUserAdjlist->end(); ++it1){
		for(std::unordered_map<int,int>::iterator it2 = it1->second->begin(); it2!=it1->second->end(); ++it2){
			if(nodeDegreeMap->count(it1->first.first)>0){
				nodeDegreeMap->at(it1->first.first) += it2->second;
				total_counts+=it2->second;
			}else{
				nodeDegreeMap->insert({it1->first.first,it2->second});
				total_counts+=it2->second;
			}
		}
	}
	// NOTE: remember to change the format of the file 
	int max_count = 0;
	while(readPiFile(piFileDescriptor, &user, s) != NULL) {
		if(getUniformRandom()>0.50)
			continue;
		if(nodeDegreeMap->count(user)>0){
			outfile<<user<<","<<nodeDegreeMap->at(user)<<s;//<<endl;
			if(max_count<nodeDegreeMap->at(user))
				max_count=nodeDegreeMap->at(user);
		}
		else
			outfile<<user<<","<<0<<s;//<<endl;
	}
	cout<<"max_count "<<max_count<<endl;
	outfile.flush();
	outfile.close();
}

char* Utils::readPiFile(FILE* graph_file_pointer, int* u1, char* s) {
	//char seq[20000];
	int err = fscanf(graph_file_pointer, "%d", u1); 
	return fgets(s, 20000, graph_file_pointer);

//	printf("%d\t%d\t%d\t%s\n", *u1, *u2, *tid, s);
}

void Utils::generateSimilarityCount(char* clusterFile, char* countFile,std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*,class_hash<pair<int,int>>>* completeUserAdjlist, int K){
	std::unordered_map<std::pair<int,int>, int, class_hash<std::pair<int,int>>>* userUserCounts = new std::unordered_map<std::pair<int,int>, int, class_hash<std::pair<int,int>>>();
	FILE* discreteClusters = fopen(clusterFile, "r");
	int user;
	int cluster;
	char s[20000];
	std::unordered_map<int, std::vector<int>*>* clusterUserMap = new std::unordered_map<int, std::vector<int>*>();
	ofstream outfile(countFile);
	while(readSeedIndexFile(discreteClusters, &user, &cluster, s) != NULL) {
//		cout<<user<<" "<<cluster;
		if(clusterUserMap->count(cluster))
			clusterUserMap->at(cluster)->push_back(user);
		else{
			clusterUserMap->insert({cluster, new std::vector<int>()});
			clusterUserMap->at(cluster)->push_back(user);
		}
	}
	cout<<"generated cluster-user map\n";
	std::vector<int>* fullSortedList = new std::vector<int>();
	for(int k=1; k<=K; ++k){
		cout<<"k is "<<k<<endl;
		fullSortedList->insert(fullSortedList->end(),clusterUserMap->at(k)->begin(), clusterUserMap->at(k)->end());
		cout<<"cluster  "<<k<<" size "<<clusterUserMap->at(k)->size()<<endl;
	}
	cout<<"final list size "<<fullSortedList->size()<<endl;
	long total_counts = 0;
	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<std::pair<int,int>>>::iterator it1=completeUserAdjlist->begin(); it1!=completeUserAdjlist->end(); ++it1){
//		if(it1->second->size()<=1)
		for(std::unordered_map<int,int>::iterator it2 = it1->second->begin(); it2!=it1->second->end(); ++it2){
			std::pair<int,int> user_pair = std::make_pair(it1->first.first, it2->first);
			if(userUserCounts->count(user_pair)>0){
				userUserCounts->at(user_pair) += it2->second;
				total_counts+=it2->second;
			}else{
				userUserCounts->insert({user_pair,it2->second});
				total_counts+=it2->second;
			}
		}
	}
	cout<<"generated userUserCounts "<<total_counts<<"\n";
	// NOTE: careful to bring the clusterfile in this fromat separated by space
//	while(readSeedIndexFile(discreteClusters, &user, &cluster, s) != NULL) {
//		cout<<user<<" "<<cluster;
//		if(clusterUserMap->count(cluster))
//			clusterUserMap->at(cluster)->push_back(user);
//		else{
//			clusterUserMap->insert({cluster, new std::vector<int>()});
//			clusterUserMap->at(cluster)->push_back(user);
//		}
//	}
	// Unravel the map into 
//	std::vector<int>* fullSortedList = new std::vector<int>();
//	for(int k=1; k<=K; ++k){
//		fullSortedList->insert(fullSortedList->end(),clusterUserMap->at(k)->begin(), clusterUserMap->at(k)->end());
//		cout<<"cluster  "<<k<<" size "<<clusterUserMap->at(k)->size()<<endl;
//	}
	for(std::vector<int>::iterator it1 = fullSortedList->begin(); it1!=fullSortedList->end(); ++it1){
		for(std::vector<int>::iterator it2 = fullSortedList->begin(); it2!=fullSortedList->end(); ++it2){
			std::pair<int, int> userPair = std::make_pair((*it1),(*it2));
			if(userUserCounts->count(userPair)>0){
				outfile<<userUserCounts->at(userPair)<<",";
//				cout<<"printed a non-zero "<<userUserCounts->at(userPair)<<endl;
			}
			else
				outfile<<0<<",";
		}
		outfile<<endl;
	}
	outfile.flush();
	outfile.close();

}


std::pair<int,int> Utils::getTheHeldoutSet(
std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*,class_hash<pair<int,int>>>* completeUserAdjlist, 
std::unordered_map< std::pair<int,int>, std::unordered_map<int, std::pair<int,int>>*,class_hash<pair<int,int>>>* heldoutUserAdjlist, 
double heldPercent, std::unordered_map<int,std::unordered_set<int>*>* perThreadUserSet, int num_users,
std::unordered_map<int,int>* userIndexMap,
std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*,class_hash<pair<int,int>>>* heldoutUserAdjlist_held,
char* filename, int attemptThreshold){

	int numHeldoutEdges = 0;
	int totalLinkEdges = 0;
//	int attemptThreshold = -1;//10;
	int testOrHeldEdges = 0;
	int testOrHeldNonEdges = 0;
	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<pair<int,int>>>::iterator it1=completeUserAdjlist->begin(); it1!=completeUserAdjlist->end(); ++it1){
//		if(it1->second->size()<=1)
		for(std::unordered_map<int,int>::iterator it2 = it1->second->begin(); it2!=it1->second->end(); ++it2){

			totalLinkEdges++;

			if(getUniformRandom()<=heldPercent){
				std::pair<int,int> user_thread = it1->first;
				int threadId = it1->first.second;
				
				if(heldoutUserAdjlist->count(user_thread)>0){
					heldoutUserAdjlist->at(user_thread)->insert({it2->first, std::make_pair(testOrHeldEdges,it2->second)});
				}else{
					heldoutUserAdjlist->insert({user_thread, new std::unordered_map<int,std::pair<int,int>>()});
					heldoutUserAdjlist->at(user_thread)->insert({it2->first, std::make_pair(testOrHeldEdges,it2->second)});
				}
				if(testOrHeldEdges==0){
					if(heldoutUserAdjlist_held->count(user_thread)>0){
						heldoutUserAdjlist_held->at(user_thread)->insert({it2->first, it2->second});
					}else{
						heldoutUserAdjlist_held->insert({user_thread, new std::unordered_map<int,int>()});
						heldoutUserAdjlist_held->at(user_thread)->insert({it2->first, it2->second});
					}
				}
				numHeldoutEdges++;
                testOrHeldEdges = (testOrHeldEdges+1)%2;
				int numAttempts = 0;

				while(numAttempts<attemptThreshold){
					cout<<"in the nonEdge prodcution code"<<endl;
					int randomIndex = rand()%num_users;
				   	int randomUserId = userIndexMap->at(randomIndex);
					if(it1->second->count(randomUserId)<=0 && perThreadUserSet->at(threadId)->count(randomUserId)<=0){
						heldoutUserAdjlist->at(user_thread)->insert({randomUserId, std::make_pair(testOrHeldEdges,0)});
						if(testOrHeldNonEdges==0){
							if(heldoutUserAdjlist_held->count(user_thread)>0){
								heldoutUserAdjlist_held->at(user_thread)->insert({randomUserId, 0});
							}else{
								heldoutUserAdjlist_held->insert({user_thread, new std::unordered_map<int,int>()});
								heldoutUserAdjlist_held->at(user_thread)->insert({randomUserId, 0});
							}
						}
						numHeldoutEdges++;
						testOrHeldNonEdges = (testOrHeldNonEdges+1)%2;
						break;
					}
					numAttempts++;
//					if(numAttempts>attemptThreshold)
//						break;
				}

				//TODO for now I am not removing it form the completeList; we will just not update this edge
//				completeUserAdjlist->
			}
//			cout<<it1->first.first<<" >< "<<it2->first<<" >< "<<it1->first.second<<" >< "<<
//				it2->second<<":\t";
//			num_nonzeros++;
		}
	}
	writeHeldoutAndTestToFile(heldoutUserAdjlist, filename);
	return make_pair(numHeldoutEdges,totalLinkEdges);
}

void Utils::writeHeldoutAndTestToFile(
		std::unordered_map< std::pair<int,int>, std::unordered_map<int, std::pair<int,int>>*,class_hash<pair<int,int>>>* heldoutUserAdjlist, 
		char* fileName ){
	// 1/0(test/held),  U1, U2, thread, count, post?
	ofstream outfile(fileName);
	for(std::unordered_map< std::pair<int,int>, std::unordered_map<int,std::pair<int,int>>*, class_hash<pair<int,int>>>::iterator it1=heldoutUserAdjlist->begin(); it1!=heldoutUserAdjlist->end(); ++it1){
		int U1 = it1->first.first;
		int threadId = it1->first.second;
		for(std::unordered_map<int,std::pair<int,int>>::iterator it2 = it1->second->begin(); it2!=it1->second->end(); ++it2){
			int U2 = it2->first;
			int count = it2->second.second;
			int testOrHeldEdges = it2->second.first;
			outfile << testOrHeldEdges <<" "<<U1<<" "<<U2<<" "<<threadId<<" "<<count<<" "<<"NOPOST"<<endl;
		}
	}
	outfile.flush();
	outfile.close();
	cout<<"heldoutAndTestSet written to "<<fileName<<endl;
}

std::pair<int,int> Utils::readHeldoutAndTest( 
		std::unordered_map< std::pair<int,int>, std::unordered_map<int, std::pair<int,int>>*,class_hash<pair<int,int>>>* heldoutUserAdjlist, 
		std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*,class_hash<pair<int,int>>>* heldoutUserAdjlist_held,
		char* fileName ){
	FILE* heldAndTestFile = fopen(fileName, "r");
	int u1;
	int u2;
	int testOrHeldEdges;
	int threadId;
	int count;
	char s[20000];
	int numHeldEdges = 0;
	int numTestEdges = 0;

	while(readHeldAndTestFile(heldAndTestFile, &u1, &u2, &testOrHeldEdges, &threadId, &count, s) != NULL) {
		std::pair<int,int> user_thread = std::make_pair(u1,threadId);
		if(heldoutUserAdjlist->count(user_thread)>0){
			heldoutUserAdjlist->at(user_thread)->insert({u2, std::make_pair(testOrHeldEdges,count)});
		}else{
			heldoutUserAdjlist->insert({user_thread, new std::unordered_map<int,std::pair<int,int>>()});
			heldoutUserAdjlist->at(user_thread)->insert({u2, std::make_pair(testOrHeldEdges,count)});
		}
		numTestEdges++;
		if(testOrHeldEdges==0){
			numHeldEdges++;
			if(heldoutUserAdjlist_held->count(user_thread)>0){
				heldoutUserAdjlist_held->at(user_thread)->insert({u2, count});
			}else{
				heldoutUserAdjlist_held->insert({user_thread, new std::unordered_map<int,int>()});
				heldoutUserAdjlist_held->at(user_thread)->insert({u2, count});
			}
		}
//		cout<<testOrHeldEdges<<" "<<u1<<" "<<u2<<" "<<" "<<threadId<<" "<<count<<endl;
	}
	return std::make_pair(numHeldEdges, numTestEdges-numHeldEdges);
}

char* Utils::readHeldAndTestFile(FILE* graph_file_pointer, int* u1, int* u2, int* testOrHeldEdges, int* threadId, int* count, char* s) {
	int err = fscanf(graph_file_pointer, "%d\t%d\t%d\t%d\t%d", testOrHeldEdges, u1, u2, threadId, count); 
	return fgets(s, 20000, graph_file_pointer);

//	printf("%d\t%d\t%d\t%s\n", *u1, *u2, *tid, s);
}

void Utils::intializePiFromIndexFile(boost::numeric::ublas::matrix<double>* gamma, std::string filename, 
		std::unordered_map<int,int>* userList){
	FILE* seed_pointer = fopen(filename.c_str(), "r");
	int u1;
	int u2;
	char s[20000];
	// asuuming that gamma is already intialized with very low values
	cout<<"\n Intialing gamma via graclus cluster\n";
	while(readSeedIndexFile(seed_pointer, &u1, &u2, s) != NULL) {
		int u_index = userList->at(u1);
		(*gamma)(u_index,u2)=1;
	}
	
}

char* Utils::readSeedIndexFile(FILE* graph_file_pointer, int* u1, int* u2, char* s) {
	//char seq[20000];
	int err = fscanf(graph_file_pointer, "%d\t%d", u1, u2); 
	return fgets(s, 20000, graph_file_pointer);

//	printf("%d\t%d\t%d\t%s\n", *u1, *u2, *tid, s);
}

/*
 * userList is a list of users
 * userAdjlist is a hasmap containing key as <user, thread> pair and value as hashmap of (users,counts) it talks to 
 * userThreadPost is a hashmap containing <user, thread> pair as key and vector<int> as wordlist of posts aggregated.
 * */

//template <class T>
void Utils::readThreadStructureFile (char* fileName, std::unordered_map<int,int>* userList, 
		std::unordered_set<int>* threadList, std::unordered_set<int>* vocabList, 
		std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* userAdjlist, 
		std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost){
	int u1;
	int u2;
	int tid;
	char s[20000];

	FILE* graph_file_pointer = fopen(fileName, "r");

    int num_users=0;

	cout<<"file name to be read from "<<fileName<<endl;

    while(readFile(graph_file_pointer, &u1, &u2, &tid, s) != NULL) {
		if(userList->count(u1)<=0){
			userList->insert({u1,num_users});
			num_users++;
		}
		if(userList->count(u2)<=0){
			userList->insert({u2,num_users});
			num_users++;
		}
		threadList->insert(tid);
		pair<int, int> user_thread = std::make_pair(u1,tid);
		// Read up network component data
//		cout<<"user thread "<<user_thread.first<<" "<<user_thread.second<<" "<<u1<<" "<<u2<<"\t::\t"<<s<<endl;
		if(userAdjlist->count(user_thread)>0){
//			cout<<"got the user tid pair for N/W "<< user_thread.first<<" "<<user_thread.second<<" "<<u1<<" "<<u2<<endl;
			std::unordered_map<int, int>* adjacencyMap = userAdjlist->at(user_thread);
			if(adjacencyMap->count(u2)>0)
				adjacencyMap->at(u2)+=1;							// increment the user1-user2 interatcion count by 1
			else
				adjacencyMap->insert({u2,1});
		}else{
//			cout<<"getting the pair for the FIRST time for N/W"<<endl;
			std::unordered_map<int, int>* adjacencyMap = new std::unordered_map<int, int>();
			adjacencyMap->insert({u2,1});
			userAdjlist->insert({user_thread,adjacencyMap});
		}
		// Read up LDA component-data
		std::vector<std::string> newWords; 
		boost::split(newWords, s, boost::is_any_of("\t "));
		if(userThreadPost->count(user_thread)>0){
//			cout<<"got the user tid pair for LDA"<< user_thread.first<<" "<<user_thread.second<<" "<<u1<<" "<<u2<<endl;
			std::vector<int>* wordList = userThreadPost->at(user_thread);
            addWords(wordList, &newWords, vocabList);
		}else{
//			cout<<"getting the pair for the FIRST time for LDA"<<endl;
			std::vector<int>* wordList = new std::vector<int>();
            addWords(wordList, &newWords, vocabList);
			userThreadPost->insert({user_thread,wordList});
		}
	}
}


void Utils::addWords(std::vector<int>* wordList, std::vector<std::string>* newWords, 
		std::unordered_set<int>* vocabList){
	for(std::vector<std::string>::iterator it=newWords->begin(); it!=newWords->end(); ++it){
		std::string word = *it; boost::algorithm::trim(word);
		if(strcmp(word.c_str(),"")){
			int wordId = atoi(word.c_str());
			wordList->push_back(wordId);
			vocabList->insert(wordId);		// Every word String is a indexId itself
		}
	}
}

char* Utils::readFile(FILE* graph_file_pointer, int* u1, int* u2, int* tid, char* s) {
	//char seq[20000];
	int err = fscanf(graph_file_pointer, "%d\t%d\t%d", u1, u2, tid); 
//	printf("%d\t%d\t%d\t%s\n", *u1, *u2, *tid, s);
	return fgets(s, 20000, graph_file_pointer);

//	printf("%d\t%d\t%d\t%s\n", *u1, *u2, *tid, s);
}

//int main(int argc, char** argv){
//	Utils* utils = new Utils();
//    utils->readThreadStructureFile<int>((string)"username_mention_graph.txt");
//	return 0; 
//}


/* The trigamma function is the derivative of the digamma function.

   Reference:

    B Schneider,
    Trigamma Function,
    Algorithm AS 121,
    Applied Statistics,
    Volume 27, Number 1, page 97-99, 1978.

    From http://www.psc.edu/~burkardt/src/dirichlet/dirichlet.f
    (with modification for negative arguments and extra precision)

    http://code.google.com/p/graphlabapi/source/browse/demoapps/clustering/gamma.cpp?name=8bbebad70f&r=46a44457777a3d2876126a916651be396ef886ca
	http://pmtksupport.googlecode.com/svn/trunk/lightspeed2.3/util.c
*/


double Utils::getUniformRandom(){
	boost::mt19937 rng;
	rng.seed(static_cast<unsigned int>(std::time(0)));
//	rng.distribution().reset();
	static boost::uniform_01<boost::mt19937> zeroone(rng);
	return zeroone();

}


float Utils::trigamma(double x)
{
  double result;
  double neginf = -1.0/0.0,
          small = 1e-4,
          large = 8,
          c = 1.6449340668482264365, /* pi^2/6 = Zeta(2) */
          c1 = -2.404113806319188570799476,  /* -2 Zeta(3) */
          b2 =  1./6,
          b4 = -1./30,
          b6 =  1./42,
          b8 = -1./30,
          b10 = 5./66;
  /* Illegal arguments */
  if((x == neginf) || std::isnan(x)) {
    return 0.0/0.0;
  }
  /* Singularities */
  if((x <= 0) && (floor(x) == x)) {
    return -neginf;
  }
  /* Negative values */
  /* Use the derivative of the digamma reflection formula:
   * -trigamma(-x) = trigamma(x+1) - (pi*csc(pi*x))^2
   */
  if(x < 0) {
    result = M_PI/sin(-M_PI*x);
    return -trigamma(1-x) + result*result;
  }
  /* Use Taylor series if argument <= small */
  if(x <= small) {
    return 1/(x*x) + c + c1*x;
  }
  result = 0;
  /* Reduce to trigamma(x+n) where ( X + N ) >= B */
  while(x < large) {
    result += 1/(x*x);
    x++;
  }
  /* Apply asymptotic formula when X >= B */
  /* This expansion can be computed in Maple via asympt(Psi(1,x),x) */
  if(x >= large) {
    double r = 1/(x*x);
    result += 0.5*r + (1 + r*(b2 + r*(b4 + r*(b6 + r*(b8 + r*b10)))))/x;
  }
  return result;
}

