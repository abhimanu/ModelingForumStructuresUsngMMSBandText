#ifndef IBM_BASE_HPP_
#define IBM_BASE_HPP_

#include <vector>
#include <boost/unordered_set.hpp>

// Are we compiling for MEX?
#ifdef _MEX
#include <mex.h>
#define IBMPRINT(str) mexPrintf(str.c_str()); mexEvalString("pause(.001);")
#else
#include <iostream>
#define IBMPRINT(str) std::cout << str
#endif

namespace ibm {

// Typedefs
typedef unsigned short uint16; // Should work for all x86/x64 compilers
typedef unsigned int   uint32; // Should work for all x86/x64 compilers
typedef std::vector<uint16> uint16Vec;
typedef std::vector<uint32> uint32Vec;
typedef std::vector<double> doubleVec;
typedef boost::unordered_set<uint32> uint32Set;

} // namespace ibm

#endif
