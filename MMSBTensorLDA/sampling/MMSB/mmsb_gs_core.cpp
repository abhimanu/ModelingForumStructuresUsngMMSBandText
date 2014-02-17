#include "Mmsb.hpp"

#include <limits>

using namespace std;

/*
 * MEX gateway function.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  /* Input argument list:
   * 0.  E (NxN logical)
   * 1.  sR (NxN uint16)
   * 2.  sL (NxN uint16)
   * 3.  alpha (scalar double)
   * 4.  lambda0 (scalar double)
   * 5.  lambda1 (scalar double)
   * 6.  K (scalar uint16)
   * 7.  modNegLinkLL (scalar logical)
   * 8.  rngSeed (scalar uint32)
   * 9.  useHpTuning (1x2 logical)
   * 10. ITERS (scalar uint32)
   * 11. NUM_INITS (scalar uint32)
   * 12. INITIALIZE (scalar logical)
   *
   * Output argument list:
   * 0.  sR (NxN uint16)
   * 1.  sL (NxN uint16)
   * 2.  alpha (scalar double)
   * 3.  lambda0 (scalar double)
   * 4.  lambda1 (scalar double)
   * 5.  ll (1xITERS double) (complete log-likelihood at every iteration)
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
  bool* useHpTuning;
  ibm::uint32 ITERS;
  ibm::uint32 NUM_INITS;
  bool INITIALIZE;
  
  double* alpha_out;
  double* lambda0_out;
  double* lambda1_out;
  double* ll;
  
  // Check number of arguments
  if (nrhs < 13 || nrhs > 13) {
    mexErrMsgTxt("Requires 13 input arguments");
  } else if (nlhs < 6 || nlhs > 6) {
    mexErrMsgTxt("Requires 6 output arguments");
  }
  
  // Initialize some outputs
  plhs[0] = mxDuplicateArray(prhs[1]);  // sR
  plhs[1] = mxDuplicateArray(prhs[2]);  // sL
  plhs[2] = mxDuplicateArray(prhs[3]);  // alpha
  plhs[3] = mxDuplicateArray(prhs[4]);  // lambda0
  plhs[4] = mxDuplicateArray(prhs[5]);  // lambda1
  
  // Get inputs/outputs/pointers
  E               = (bool*) mxGetPr(prhs[0]);
  sR              = (ibm::uint16*) mxGetPr(plhs[0]);
  sL              = (ibm::uint16*) mxGetPr(plhs[1]);
  alpha_out       = (double*) mxGetPr(plhs[2]);
  lambda0_out     = (double*) mxGetPr(plhs[3]);
  lambda1_out     = (double*) mxGetPr(plhs[4]);
  
  alpha           = (double) mxGetScalar(prhs[3]);
  lambda0         = (double) mxGetScalar(prhs[4]);
  lambda1         = (double) mxGetScalar(prhs[5]);
  N               = (ibm::uint32) mxGetM(prhs[0]);  // Determine N from variable E
  K               = (ibm::uint16) mxGetScalar(prhs[6]);
  modNegLinkLL    = (bool) mxGetScalar(prhs[7]);
  rngSeed         = (ibm::uint32) mxGetScalar(prhs[8]);
  useHpTuning     = (bool*) mxGetPr(prhs[9]);
  ITERS           = (ibm::uint32) mxGetScalar(prhs[10]);
  NUM_INITS       = (ibm::uint32) mxGetScalar(prhs[11]);
  INITIALIZE      = (bool) mxGetScalar(prhs[12]);
  
  // Initialize remaining outputs
  plhs[5]         = mxCreateNumericMatrix(1,ITERS,mxDOUBLE_CLASS,0);
  ll		          = (double*) mxGetPr(plhs[5]);
  
  // Error checking
  if (mxGetM(prhs[9]) != 1 || mxGetN(prhs[9]) != 2) mexErrMsgTxt("useHpTuning must be 1x2");
  
  if (!mxIsClass(prhs[0],"logical")) mexErrMsgTxt("E must be logical");
  if (!mxIsClass(plhs[0],"uint16")) mexErrMsgTxt("sR must be uint16");
  if (!mxIsClass(plhs[1],"uint16")) mexErrMsgTxt("sL must be uint16");
  if (mxGetM(prhs[0]) != N || mxGetN(prhs[0]) != N) mexErrMsgTxt("E must be NxN");
  if (mxGetM(plhs[0]) != N || mxGetN(plhs[0]) != N) mexErrMsgTxt("sR must be NxN");
  if (mxGetM(plhs[1]) != N || mxGetN(plhs[1]) != N) mexErrMsgTxt("sL must be NxN");
  
  if (alpha <= 0) mexErrMsgTxt("alpha must be >0");
  if (lambda0 <= 0) mexErrMsgTxt("lambda0 must be >0");
  if (lambda1 <= 0) mexErrMsgTxt("lambda1 must be >0");
  if (NUM_INITS == 0) mexErrMsgTxt("NUM_INITS must be >0");
  
  // Initialize Gibbs sampler
  ibm::MmsbSampler sampler(E,sR,sL,alpha,lambda0,lambda1,
                           N,K,modNegLinkLL,rngSeed);
  if (INITIALIZE) { // Run generative process, which also initializes the sufficient statistics
    // Run generative process, taking the best out of NUM_INITS initializations
    mexPrintf("......Initializing with generative process, best of %d initializations\n",NUM_INITS);
    mexEvalString("pause(.001);");
    ibm::uint16Vec best_sR(N*N);
    ibm::uint16Vec best_sL(N*N);
    double best_ll = -std::numeric_limits<double>::infinity();
    for (unsigned int t = 0; t < NUM_INITS; ++t) {
      sampler.generativeLatent();
      sampler.initializeSSObserved();
      double cur_ll = sampler.computeCompleteLogLikelihood();
      mexPrintf("......%d: ll = %f\n",t+1,cur_ll);
      mexEvalString("pause(.001);");
      if (cur_ll > best_ll) {
        // Copy current initialization
        best_ll = cur_ll;
        for (size_t i = 0; i < N*N; ++i) {
          best_sR[i] = sR[i];
          best_sL[i] = sL[i];
        }
      }
      sampler.clearSS();
    }
    
    // Restore best initialization
    mexPrintf("......Best initialization ll = %f\n",best_ll);
    mexEvalString("pause(.001);");
    for (size_t i = 0; i < N*N; ++i) {
      sR[i] = best_sR[i];
      sL[i] = best_sL[i];
    }
    sampler.initializeSS();
  } else {  // Don't run generative process, just initialize sufficient statistics with input
    mexPrintf("......Initializing with latent variable inputs\n");
    mexEvalString("pause(.001);");
    sampler.initializeSS();
  }
  
  // Run sampler for ITERS iterations
  for (ibm::uint32 t = 0; t < ITERS; ++t) {
    if ((t+1) % 100 == 0) {
      mexPrintf("......Iteration %d\n",t+1);
      mexEvalString("pause(.001);");
    }
    sampler.gs();
    if (useHpTuning[0]) {
      sampler.mhAlpha();
    }
    if (useHpTuning[1]) {
      sampler.mhLambda();
    }
    ll[t] = sampler.computeCompleteLogLikelihood();
  }
  
  // Output hyperparameter estimates
  alpha_out[0]    = sampler.getAlpha();
  lambda0_out[0]  = sampler.getLambda0();
  lambda1_out[0]  = sampler.getLambda1();
}
