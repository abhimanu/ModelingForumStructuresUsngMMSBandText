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

/*
 * userList is a list of users
 * userAdjlist is a hasmap containing key as <user, thread> pair and value as hashmap of (users,counts) it talks to 
 * userThreadPost is a hashmap containing <user, thread> pair as key and vector<int> as wordlist of posts aggregated.
 * */

//template <class T>
void Utils::readThreadStructureFile (std::string fileName, std::unordered_set<int>* userList, 
		std::unordered_set<int>* threadList,
		std::unordered_map< std::pair<int,int>, std::unordered_map<int, int>*, class_hash<pair<int,int>>>* userAdjlist, 
		std::unordered_map< std::pair<int,int>, std::vector<int>*, class_hash<pair<int,int>>>* userThreadPost){
	int u1;
	int u2;
	int tid;
	char s[20000];

	FILE* graph_file_pointer = fopen(fileName.c_str(), "r");
	

    while(readFile(graph_file_pointer, &u1, &u2, &tid, s) != NULL) {
		userList->insert(u1);
		userList->insert(u2);
		threadList->insert(tid);
		pair<int, int> user_thread = std::make_pair(u1,tid);
		// Read up network component data
		cout<<"user thread "<<user_thread.first<<" "<<user_thread.second<<" "<<u1<<" "<<u2<<"\t::\t"<<s<<endl;
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
            addWords(wordList, &newWords);
		}else{
//			cout<<"getting the pair for the FIRST time for LDA"<<endl;
			std::vector<int>* wordList = new std::vector<int>();
            addWords(wordList, &newWords);
			userThreadPost->insert({user_thread,wordList});
		}
	}
}


void Utils::addWords(std::vector<int>* wordList, std::vector<std::string>* newWords){
	for(std::vector<std::string>::iterator it=newWords->begin(); it!=newWords->end(); ++it){
		std::string word = *it; boost::algorithm::trim(word);
		if(strcmp(word.c_str(),"")){
		int wordId = atoi(word.c_str());
		wordList->push_back(wordId);
		}
	}
}

char* Utils::readFile(FILE* graph_file_pointer, int* u1, int* u2, int* tid, char* s) {
	//char seq[20000];
	int err = fscanf(graph_file_pointer, "%d\t%d\t%d", u1, u2, tid); 
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

