/*
 * Utils.cpp
 *
 *  Created on: Apr 26, 2013
 *      Author: abhimank
 */

#include <boost/numeric/ublas/matrix.hpp>
#include "csv_parser.hpp"
#include "Utils.h"

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
  if((x == neginf) || isnan(x)) {
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

