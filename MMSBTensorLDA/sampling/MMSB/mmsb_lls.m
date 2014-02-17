% Wrapper for the MMSB C++ marginal log likelihood sampler.
% Because MATLAB uses column-major indexing but C++ uses row-major indexing,
% we transpose all input and output matrixes to the MEX function.
function ll = mmsb_lls(E,alpha,lambda0,lambda1,K,modNegLinkLL,rngSeed,ITERS)
  % Ensure all inputs have the correct type
  E             = logical(E');  % Switch to C++ row-major indexing
  K             = uint16(K);
  modNegLinkLL  = logical(modNegLinkLL);
  rngSeed       = uint32(rngSeed);
  ITERS         = uint32(ITERS);
  % Run marginal log likelihood sampler
  ll = mmsb_lls_core(E,alpha,lambda0,lambda1,K,modNegLinkLL,rngSeed,ITERS);
end
