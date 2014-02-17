#include "Mmsb.hpp"

#include <limits>

using namespace std;

/*
 * MEX gateway function.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  /* Input argument list:
   * 0.  alpha (scalar double)
   * 1.  lambda0 (scalar double)
   * 2.  lambda1 (scalar double)
   * 3.  N (scalar uint32)
   * 4.  K (scalar uint16)
   * 5.  rngSeed (scalar uint32)
   *
   * Output argument list:
   * 0.  E (NxN logical)
   * 1.  sR (NxN uint16)
   * 2.  sL (NxN uint16)
   */
  
  bool* E;
  ibm::uint16* sR;
  ibm::uint16* sL;
  double alpha;
  double lambda0;
  double lambda1;
  ibm::uint32 N;
  ibm::uint16 K;
  ibm::uint32 rngSeed;
  
  // Check number of arguments
  if (nrhs < 6 || nrhs > 6) {
    mexErrMsgTxt("Requires 6 input arguments");
  } else if (nlhs < 3 || nlhs > 3) {
    mexErrMsgTxt("Requires 3 output arguments");
  }
  
  // Get inputs/outputs/pointers
  N               = (ibm::uint32) mxGetScalar(prhs[3]);
  K               = (ibm::uint16) mxGetScalar(prhs[4]);
  alpha           = (double) mxGetScalar(prhs[0]);
  lambda0         = (double) mxGetScalar(prhs[1]);
  lambda1         = (double) mxGetScalar(prhs[2]);
  rngSeed         = (ibm::uint32) mxGetScalar(prhs[5]);
  
  // Error chicking
  if (alpha <= 0) mexErrMsgTxt("alpha must be >0");
  if (lambda0 <= 0) mexErrMsgTxt("lambda0 must be >0");
  if (lambda1 <= 0) mexErrMsgTxt("lambda1 must be >0");
  
  // Initialize outputs
  plhs[0] = mxCreateLogicalMatrix(N,N);                   // E
  plhs[1] = mxCreateNumericMatrix(N,N,mxUINT16_CLASS,0);  // s_r
  plhs[2] = mxCreateNumericMatrix(N,N,mxUINT16_CLASS,0);  // s_l
  E       = (bool*) mxGetPr(plhs[0]);
  sR      = (ibm::uint16*) mxGetPr(plhs[1]);
  sL      = (ibm::uint16*) mxGetPr(plhs[2]);
  
  // Initialize and run generative process
  ibm::MmsbSampler sampler(E,sR,sL,alpha,lambda0,lambda1,
                           N,K,false,rngSeed);
  sampler.generative();
}
