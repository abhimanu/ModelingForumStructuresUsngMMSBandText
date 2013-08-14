/*
 * Utils.h
 *
 *  Created on: Apr 26, 2013
 *      Author: abhimank
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <boost/numeric/ublas/matrix.hpp>
#include "csv_parser.hpp"
#include <boost/tr1/functional.hpp>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

template <class T>
class class_hash{
	const size_t bucket_size = 10; //
	const size_t min_buckets = (1<<10);

public:

	size_t operator()(const T& key_value) const{
		hash<std::string> hash_fn;
		string keyVal = std::to_string(key_value.first) + std::to_string(key_value.second);
//		std::cout<<keyVal<<" "<<hash_fn(keyVal)<<" "<<hash_fn(keyVal)<<std::endl;
		return hash_fn(keyVal);
	}

    bool operator()(const T& left, const T& right) const{
		return (((int)left.first < (int)right.first)||
		 (((int)left.first == (int)right.first)&&((int)left.first < (int)right.first)));
	}
};

class Utils {

public:
	boost::numeric::ublas::matrix<int>* readCsvToMat(char* filename, int numRows, int numColumns);
	//Utils();
	//virtual ~Utils();
	float trigamma(double x);
	
	void addWords(std::vector<int>* wordList, std::vector<std::string>* newWords);

//	template <class T>
	void readThreadStructureFile (std::string fileName, std::unordered_map<int,int>* userList, 
			std::unordered_set<int>* threadList,
			std::unordered_map< std::pair<int,int>, std::unordered_map<int,int>*, class_hash<std::pair<int,int>>>* userAdjlist ,
			std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<std::pair<int,int>>>* userThreadPost);

	void getTheHeldoutSet(std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* completeUserAdjlist, std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* heldoutUserAdjlist, double heldPercent);
	
	char* readFile(FILE* graph_file_pointer, int* u1, int* u2, int* tid, char* s);

	double getUniformRandom();
};

#endif /* UTILS_H_ */
