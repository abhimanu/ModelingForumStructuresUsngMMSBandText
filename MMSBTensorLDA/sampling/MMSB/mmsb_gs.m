% Wrapper for the MMSB C++ Gibbs sampler.
% Because MATLAB uses column-major indexing but C++ uses row-major indexing,
% we transpose all input and output matrixes to the MEX function.
function samples = mmsb_gs(E,alpha,lambda0,lambda1,K,modNegLinkLL, ...
                           rngSeed,useHpTuning,GS_PARAM,resumefile)
  % Ensure all inputs have the correct type
  E             = logical(E');  % Switch to C++ row-major indexing
  K             = uint16(K);
  modNegLinkLL  = logical(modNegLinkLL);
  rngSeed       = uint32(rngSeed);
  useHpTuning   = logical(useHpTuning);
  
  if exist('resumefile','var') && ~isempty(resumefile) && exist(resumefile,'file')
    % Resume from saved state
    fprintf('Resuming state from %s...\n',resumefile);
    load(resumefile);
  else
    % Initialize parameters
    N = size(E,1);
    % Declare return variables
    samples                 = cell(1,GS_PARAM.NUM_SAMPLES - GS_PARAM.BURN_IN);
    samples{end}.all_ll     = [];
    samples{end}.gs_time    = 0;
    samples{end}.rngSeed    = rngSeed;
    % Set iteration counter
    iter                    = uint32(0);
  end
  
  % Take samples
  while iter < GS_PARAM.NUM_SAMPLES
    gs_timer  = tic;
    iter      = iter + 1;
    if iter == 1
      % First iteration initialization
      % Use generative process in topicblock_gs_core to initialize;
      % for now just declare empty inputs.
      sR          = uint16(zeros(N,N));
      sL          = uint16(zeros(N,N));
      INITIALIZE  = true;
    else
      INITIALIZE  = false;  % Initialize using last iteration's results
    end
    LAG = uint32(GS_PARAM.LAG);
    
    if iter <= GS_PARAM.BURN_IN
      fprintf('...Burn-in %d/%d (total samples to take %d, lag for %d iterations)',iter,GS_PARAM.BURN_IN,GS_PARAM.NUM_SAMPLES,LAG);
    else
      fprintf('...Taking sample %d/%d (lag for %d iterations)',iter,GS_PARAM.NUM_SAMPLES,LAG);
    end
    
    HP_MH_SETTING   = useHpTuning;
    if iter <= GS_PARAM.NOMH_SAMPLES
      HP_MH_SETTING(:)  = false;    % Don't use hyperparameter MH too early
    end
    if ~any(HP_MH_SETTING)
      fprintf(' ...Hyperparameter MH is OFF\n');
    else
      fprintf(' ...Hyperparameter MH is ON\n');
    end
    
    [sR,sL,alpha,lambda0,lambda1,ll] = mmsb_gs_core(E,sR,sL,alpha,lambda0,lambda1,K,modNegLinkLL, ...
                                                    rngSeed+iter-1,HP_MH_SETTING,LAG+1, ...
                                                    uint32(GS_PARAM.NUM_INITS),INITIALIZE);
    
    % Save the sample only after burn-in
    if iter > GS_PARAM.BURN_IN
      i                   = iter - GS_PARAM.BURN_IN;
      samples{i}.sR       = sR';      % Switch to MATLAB column-major indexing
      samples{i}.sL       = sL';
      samples{i}.alpha    = alpha;    % Hyperparameters will change depending on hyperparameter_mh_flags
      samples{i}.lambda0  = lambda0;
      samples{i}.lambda1  = lambda1;
      samples{i}.ll       = ll;       % Complete log likelihoods for every lag iteration, for this sample
    end
    
    % Save sampler runtime and complete log likelihood trajectory for all samples
    samples{end}.all_ll   = [samples{end}.all_ll ll];
    samples{end}.gs_time  = samples{end}.gs_time + toc(gs_timer);
    
    % Save state
    if exist('resumefile','var') && ~isempty(resumefile)
      % Save to a tempfile first, then rename (just in case we get interrupted)
      fprintf('Saving state to %s...\n',resumefile);
      tempfile    = [resumefile '.temp'];
      save(tempfile);
      movefile(tempfile,resumefile);
    end
  end
end
