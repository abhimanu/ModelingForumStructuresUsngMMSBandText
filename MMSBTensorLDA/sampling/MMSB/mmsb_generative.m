% Wrapper for the MMSB C++ generative process sampler.
% Because MATLAB uses column-major indexing but C++ uses row-major indexing,
% we transpose all input and output matrixes to the MEX function.
function data = mmsb_generative(alpha,lambda0,lambda1,N,K,rngSeed)
  % Ensure all inputs have the correct type
  N             = uint32(N);
  K             = uint16(K);
  rngSeed       = uint32(rngSeed);
  % Run generative process
  [data.E,data.sR,data.sL] = mmsb_generative_core(alpha,lambda0,lambda1,N,K,rngSeed);
  data.E  = data.E';  % Switch to MATLAB column-major indexing
  data.sR = data.sR';
  data.sL = data.sL';
end
