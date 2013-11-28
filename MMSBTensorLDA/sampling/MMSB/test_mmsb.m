% MMSB test function

function data = test_mmsb()
  % Model parameters
  N             = 10;
  K             = 3;
  alpha         = 0.1;
  lambda0       = 0.1;
  lambda1       = 0.1;
  modNegLinkLL  = false;
  useHpTuning   = [true true];
  
  % Marginal log-likelihood sampler parameters
  SAMPLES_lls             = 1000;
  
  % Gibbs sampling parameters
  GS_PARAM.NUM_SAMPLES    = 20;
  GS_PARAM.BURN_IN        = 10;
  GS_PARAM.NOMH_SAMPLES   = 5;
  GS_PARAM.LAG            = 9;
  GS_PARAM.NUM_INITS      = 10;
  
  % Random seeds
  SEED_gen                = 12345; %sum(100*clock);
  SEED_lls                = 22345; %sum(100*clock);
  SEED_gs                 = 32345; %sum(100*clock);
  
  % Run generative process
  fprintf('Running generative process...');
  gen_start       = tic;
  data.generative = mmsb_generative(alpha,lambda0,lambda1,N,K,SEED_gen);
  data.gen_time   = toc(gen_start);
  fprintf(' finished in %f seconds\n',data.gen_time);
  
  % Record parameters
  data.E                  = data.generative.E;
  data.N                  = N;
  data.K                  = K;
  data.alpha              = alpha;
  data.lambda0            = lambda0;
  data.lambda1            = lambda1;
  data.modNegLinkLL       = modNegLinkLL;
  data.useHpTuning        = useHpTuning;
  data.SAMPLES_lls        = SAMPLES_lls;
  data.GS_PARAM           = GS_PARAM;
  data.SEED_gen           = SEED_gen;
  data.SEED_lls           = SEED_lls;
  data.SEED_gs            = SEED_gs;
  
  % Run marginal log-likelihood sampler
  fprintf('Running marginal log-likelihood sampler...\n');
  lls_start       = tic;
  data.ll         = mmsb_lls(data.E,alpha,lambda0,lambda1,K,modNegLinkLL,SEED_lls,SAMPLES_lls);
  data.lls_time   = toc(lls_start);
  fprintf('Marginal log-likelihood = %f, finished in %f seconds\n',data.ll,data.lls_time);
  
  % Run Gibbs sampling algorithm
  fprintf('Running Gibbs sampler...\n');
  gs_start        = tic;
  data.samples    = mmsb_gs(data.E,alpha,lambda0,lambda1,K,modNegLinkLL, ...
                            SEED_gs,useHpTuning,GS_PARAM);
  data.gs_time    = toc(gs_start);
  fprintf('Finished Gibbs sampling in %f seconds\n',data.gs_time);
  
  % Run heldout Gibbs sampling algorithm
  % fprintf('Running heldout Gibbs sampler...\n');
  % heldout_gs_start        = tic;
  % data.heldout_samples    = topicblock_heldout_gs(V,M_offsets,data.w,data.E, ...
                                                  % data.samples{end}.gamma,data.samples{end}.alpha,data.samples{end}.eta, ...
                                                  % data.samples{end}.lambda1,data.samples{end}.lambda2, ...
                                                  % GS_PARAM,SEED_gs,NUM_THREADS_gs, ...
                                                  % true,true, ...
                                                  % data.samples{end}.r,data.samples{end}.z,data.samples{end}.s_r,data.samples{end}.s_l, ...
                                                  % ceil(N/2));
  % data.heldout_gs_time    = toc(heldout_gs_start);
  % fprintf('Finished heldout Gibbs sampling in %f seconds\n',data.heldout_gs_time);
  
  % Run heldout_nonewpaths GS algorithm
  % fprintf('Running heldout_nonewpaths Gibbs sampler...\n');
  % heldout_nnp_gs_start        = tic;
  % data.heldout_nnp_samples    = topicblock_heldout_nonewpaths_gs(V,M_offsets,data.w,data.E, ...
                                                  % data.samples{end}.gamma,data.samples{end}.alpha,data.samples{end}.eta, ...
                                                  % data.samples{end}.lambda1,data.samples{end}.lambda2, ...
                                                  % GS_PARAM,SEED_gs,NUM_THREADS_gs, ...
                                                  % true,true, ...
                                                  % data.samples{end}.r,data.samples{end}.z,data.samples{end}.s_r,data.samples{end}.s_l, ...
                                                  % ceil(N/2));
  % data.heldout_nnp_gs_time    = toc(heldout_nnp_gs_start);
  % fprintf('Finished heldout_nonewpaths Gibbs sampling in %f seconds\n',data.heldout_nnp_gs_time);
end
