/*
 * SimpleLDA.h
 *
 *  Created on: Apr 9, 2013
 *      Author: abhimank
 */

#ifndef SIMPLELDA_H_
#define SIMPLELDA_H_

#include<string>

class SimpleLDA {
public:
	SimpleLDA();
	virtual ~SimpleLDA();
	void initialize(string* filenamesRead, int K, int sampleIters, int outer, int L, int assigned_users, string filenameSave);

private:


};

#endif /* SIMPLELDA_H_ */
