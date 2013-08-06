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

class Utils {

public:
	boost::numeric::ublas::matrix<int>* readCsvToMat(char* filename, int numRows, int numColumns);
	//Utils();
	//virtual ~Utils();
	float trigamma(double x);


};

#endif /* UTILS_H_ */
