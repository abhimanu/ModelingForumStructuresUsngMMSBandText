#include "Mmsb.hpp"

#include <limits>

using namespace std;

/*
 * MEX gateway function.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  /* Input argument list:
   * 0.  E (NxN logical)
   * 1.  alpha (scalar double)
   * 2.  lambda0 (scalar double)
   * 3.  lambda1 (scalar double)
   * 4.  K (scalar uint16)
   * 5.  modNegLinkLL (scalar logical)
   * 6.  rngSeed (scalar uint32)
   * 7.  ITERS (scalar uint32)
   *
   * Output argument list:
   * 0.  ll (scalar double)
   */
  
  bool* E;
  ibm::uint16* sR;
  ibm::uint16* sL;
  double alpha;
  double lambda0;
  double lambda1;
  ibm::uint32 N;
  ibm::uint16 K;
  bool modNegLinkLL;
  ibm::uint32 rngSeed;
  ibm::uint32 ITERS;
  double* ll;
  
  // Check number of arguments
  if (nrhs < 8 || nrhs > 8) {
    mexErrMsgTxt("Requires 8 input arguments");
  } else if (nlhs < 1 || nlhs > 1) {
    mexErrMsgTxt("Requires 1 output argument");
  }
  
  // Get inputs/outputs/pointers
  E               = (bool*) mxGetPr(prhs[0]);
  alpha           = (double) mxGetScalar(prhs[1]);
  lambda0         = (double) mxGetScalar(prhs[2]);
  lambda1         = (double) mxGetScalar(prhs[3]);
  N               = (ibm::uint32) mxGetM(prhs[0]);  // Determine N from variable E
  K               = (ibm::uint16) mxGetScalar(prhs[4]);
  modNegLinkLL    = (bool) mxGetScalar(prhs[5]);
  rngSeed         = (ibm::uint32) mxGetScalar(prhs[6]);
  ITERS           = (ibm::uint32) mxGetScalar(prhs[7]);
  
  // Error checking
  if (!mxIsClass(prhs[0],"logical")) mexErrMsgTxt("E must be logical");
  if (mxGetM(prhs[0]) != N || mxGetN(prhs[0]) != N) mexErrMsgTxt("E must be NxN");
  
  if (alpha <= 0) mexErrMsgTxt("alpha must be >0");
  if (lambda0 <= 0) mexErrMsgTxt("lambda0 must be >0");
  if (lambda1 <= 0) mexErrMsgTxt("lambda1 must be >0");
  
  // Initialize temp storage
  sR  = new ibm::uint16[N*N];
  sL  = new ibm::uint16[N*N];
  
  // Initialize outputs
  const double infty  = numeric_limits<double>::infinity();
  plhs[0] = mxCreateDoubleScalar(-infty); // ll
  ll      = (double*) mxGetPr(plhs[0]);
  
  // Initialize and run marginal log-likelihood sampler for ITERS iterations
  ibm::MmsbSampler sampler(E,sR,sL,alpha,lambda0,lambda1,
                           N,K,modNegLinkLL,rngSeed);
  for (unsigned int t = 0; t < ITERS; ++t) {
      if ((t+1) % 100 == 0) {
          mexPrintf("......Iteration %d\n",t+1);
          mexEvalString("pause(.001);");
      }
      double ll_sample    = sampler.mllSample();
      if (ll_sample > -infty) { // Only update if exp(ll_sample) > 0
          // Update ll as a running sum. Before exponentiation, normalize
          // ll_sample and ll by their max to prevent over/underflow.
          double normalizer = max(ll_sample,*ll);
          *ll               = log(exp(ll_sample-normalizer) + exp(*ll-normalizer))
                              + normalizer;
      }
  }
  *ll -= log(static_cast<double>(ITERS));  // Compute the log average marginal likelihood
  
  // Cleanup
  delete [] sR;
  delete [] sL;
}
