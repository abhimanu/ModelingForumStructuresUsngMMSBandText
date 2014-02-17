/*
 * test.cpp
 *
 *  Created on: Feb 22, 2013
 *      Author: abhimank
 */

#include <iostream>
#include <string>
#include <stdio.h>
//#include<mpi.h>
//#include <boost/numeric/ublas/vector.hpp>
//#include <boost/numeric/ublas/io.hpp>
//#include <boost/numeric/ublas/matrix.hpp>
//#include <boost/numeric/ublas/io.hpp>
#include "armadillo"
#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>

//g++ -I /usr/include   -O2   -o example1  example1.cpp  -larmadillo
//g++ -I /usr/include   -O2   -o example2  example2.cpp  -larmadillo

using namespace arma;
using namespace std;
using namespace boost::math;

fmat vectorizeFloat(fmat input, float (*function)(float)){
	fmat result(1,input.n_elem);
	for(int i=0;i<input.n_elem;i++){
		result(i)=(* function)(input(i));
	}
	return result;
}

class BaseClass{
public:
	int testBase = 20;
};

class MasterClass: public BaseClass{
public:
	int testMiscMaster = 21;
	int Master = 40;

};

class SlaveClass: public BaseClass{
public:
	int testMiscSlave = 41;
	int Slave = 60;

};

float randomGen(void){
	boost::mt19937 rng;
	static boost::uniform_01<boost::mt19937> zeroone(rng);
	return zeroone();;
}

int main(){
//	cube Nkk =ones<cube>(3,3,2);
//	cout<<Nkk;
//	Nkk.slice(0) = zeros<mat>(3,3);
//	cout<<Nkk;
//	Nkk.slice(0) = Nkk.slice(0) + ones<mat>(3,3)*3;
//	cout<<Nkk;
//	cout<<Nkk(17)<<"hello\n";
	fmat ll(1,5);
	ll.fill(5);
	ll(3)=10;
	cout<<ll;
//	fmat matlab_file;
//	matlab_file.load("monkLK.csv");
//	cout<<string("\nmonkLK.mat").append(" hehe\n")<<matlab_file<<endl;
	cout<<vectorizeFloat(ll, lgamma);
	cout<<lgamma(ll(3));
//	cout<<sizeof(float);
//	cout<<sizeof(double);
//	cout<<sizeof(int)<<endl;
//	cout<<randomGen()<<endl<<randomGen()<<endl;
//	uvec indices = find(cumsum(ll,1)>20);
//	cout<<cumsum(ll,1)<<indices(0)<<endl;
//	cout<<ceil(13.0/10)<<endl;
	cout<<repmat(ll,3,1)<<endl;
	cout<<sum(repmat(ll,3,1), 1)<<endl;
//	cout<<int(10.0)<<endl;
//	cout<<matlab_file.Base<< matlab_file.n_elem<<endl;
//	matlab_file.memptr();

//	// Test to see whether simple pointer operations would work or not
	umat* test_mat = new zeros<umat>(3,3);
	for(int i=0; i<test_mat->n_rows;i++)
		for(int j=0; j<test_mat->n_cols; j++)
			test_mat(i,j)=i*test_mat->n_cols+j;
	cout<<test_mat<<endl;
//	cout<<find(test_mat>5)<<endl;
	uvec colVec = find(*test_mat>5);
	uvec cols = floor(colVec/test_mat->n_cols);
	uvec rows = colVec-cols*test_mat->n_cols;
	uvec newColVec = rows*test_mat->n_cols+cols;
	cout<<colVec<<"\n"<<cols<<"\n"<<rows<<endl;
//	test_mat.resize(test_mat.n_elem,1);
	umat new_mat = test_mat->elem(colVec);
	cout<< test_mat->elem(colVec)<< endl;
	cout<<find(new_mat<8)<<endl;
	cout<<colVec.elem(find(new_mat<8))<<endl;
	cout<<newColVec<<endl;
	colVec(2)=0;
	cout<< test_mat->elem(colVec)<<endl;

//	unsigned int * p=&test_mat(0);
//	for(int i=0; i<test_mat.n_rows;i++){
//		for(int j=0; j<test_mat.n_cols; j++){
//			cout<<*p<<" "; p++;
//		}
//		cout<<endl;
//	}

//	struct test_struct{
//		int ms_tag;
//		float *ptr;
//	};

	BaseClass *slave = new SlaveClass();
	BaseClass *master = new MasterClass();

	cout<<"((SlaveClass*)slave)->testSlave "<<((SlaveClass*)slave)->Slave<<endl;
	cout<<"((MasterClass*)slave)->testMaster "<<((MasterClass*)slave)->Master<<endl;
	cout<<"((SlaveClass*)master)->testSlave "<<((SlaveClass*)master)->Slave<<endl;
	cout<<"((MasterClass*)master)->testMaster "<<((MasterClass*)master)->Master<<endl;
	cout<<"slave->testBase "<<slave->testBase<<endl;
	cout<<"master->testBase "<<master->testBase<<endl;

	cout<<log(10e-323)<<endl;

//	cout<<*(p)<<*(p++)<<*(p++)<<test_mat(test_mat.n_elem-1)<<endl;

}
