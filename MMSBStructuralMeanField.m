function[pi,B,alpha] = MMSBStructuralMeanField(Y, K, maxit, inner_iter)
N = size(Y,1);
% gamma = repmat(ones(1,K)*2*N/K, N, 1);

alpha = 0.5+(rand(1,K)-0.5)*0.1
B = eye(K).*0.99+1e-6;
log(B)
log(1-B)
%rB = rand(K,K)
gamma = repmat(alpha,N,1)+(rand(N,K)-0.5).*0.1;
%gamma = initGammaTrue();
% gamma = abs((rand(N,K)-0.5)*N)
%  gamma = initializeGamma(N,K)
% gamma(gamma<=0)=0.1;
% gamma
ll_iters = [];

for iter=1:maxit
    [gamma,B,ll] = updateGammasAndBlockMat(gamma,B,alpha,Y,inner_iter);
    alpha = updateAlpha(gamma,alpha);
    ll_iters = [ll_iters ll];
end


plot(1:size(ll_iters,2) , ll_iters);

pi = gamma./repmat(sum(gamma,2),1,K);
end

function gamma = initializeGamma(N,K)
gamma = rand(N,K)*N;
for i=1:N
    ind = ceil(rand*K);
    gamma(i,ind)=gamma(i,ind)+N;
end
end


function gamma = initGammaTrue()
   gamma=[11.3074    0.8397   19.1703    4.6589;
    7.1052    0.7119   27.3615    0.7978;
   33.8192    0.6972    0.7349    0.7250;
    0.6591   32.5837    0.7555    1.9780;
    0.6208   33.8553    0.7023    0.7979;
    0.6634   21.7686    0.7739   12.7704;
    0.6769    0.7856   33.6273    0.8864;
    0.6643   18.5919    0.7902   15.9299;
    0.6653   23.1739    3.9250    8.2122;
    0.7685   15.1645    0.8144   19.2289;
    0.6665   19.7664    1.2624   14.2810;
    0.6221    0.8779   33.7169    0.7594;
   10.5793    7.4841    3.9887   13.9242;
    0.7578    3.4914   14.9526   16.7746;
    0.6694    0.8183   19.5699   14.9186;
    0.6729    0.8051   16.3621   18.1363;
   33.5722    0.7335    0.8009    0.8698;
   33.6886    0.7254    0.7739    0.7885];
end

function [gamma, B, ll] = updateGammasAndBlockMat(gamma,B,alpha,Y,inner_iter)
N = size(Y,1);
K = size(alpha,2);
phi = zeros(K,K,N,N);%.*(1.0/(K*K)); %            Note that it is K><K

ll=0;
for i=1:inner_iter
for p=1:N
%     gamma_p = alpha;
    for q=1:N
        % get new phi_pq
%        temp_phi1 = variationalUpdatesPhi(Y(p,q),phi(2,p,q),B,1);
        temp_phi=variationalUpdatesPhi(Y(p,q),phi(:,:,p,q),B,gamma(p,:),gamma(q,:),alpha,inner_iter);
        phi(:,:,p,q) = temp_phi;

        
    end
    % update gamma
         gamma(p,:) = updateGamma(phi,alpha,p,N);

        % update B
          B=updateB(phi,Y,K);              % write a more efficient code for updates

%     alpha = updateAlpha(gamma,alpha);
%     ll = getLogLikelihood(gamma,B,alpha,Y,phi)
%      pi = gamma./repmat(sum(gamma,2),1,K)
%     gamma(p,:) = gamma_p;
%     B=updateB(phi,Y,K);              % write a more efficient code for updates

end
for p=1:N
     gamma(p,:) = updateGamma(phi,alpha,p,N);
     B=updateB(phi,Y,K);
end
      ll = getLogLikelihood(gamma,B,alpha,Y,phi)
%       pi = gamma./repmat(sum(gamma,2),1,K)
end
%ll

end

function ll = getLogLikelihood(gamma,B,alpha,Y,phi)
N = size(Y,1);
K = size(alpha,2);
ll=0;

% I am not sure how squeeze function works in case of 4D array

ll = ll + N*(gammaln(sum(alpha)) - sum(gammaln(alpha)));                    % line 4
for p=1:N
    deriv_phi_p = psi(gamma(p,:)) - repmat(psi(sum(gamma(p,:))),1,K);
    ll = ll + sum((alpha-1).*deriv_phi_p);                                  % line 4
    
    ll = ll - gammaln(sum(gamma(p,:))) + sum(gammaln(gamma(p,:)));          % line 5
    ll = ll - sum((gamma(p,:)-1).*deriv_phi_p);                             % line 5

    for q=1:N
        % calculate log-likelihoods
        f_B = Y(p,q)*log(B) + (1-Y(p,q))*log(1-B);
        ll = ll + sum(sum(squeeze(phi(:,:,p,q)) .* f_B));                   % line 1
        ll = ll + sum(sum(squeeze(phi(:,:,p,q)),2)'.*deriv_phi_p);                          % line 2
        
        deriv_phi_q = psi(gamma(q,:)) - repmat(psi(sum(gamma(q,:))),1,K);
        ll = ll + sum(sum(squeeze(phi(:,:,p,q)),1).*deriv_phi_q);                           % line 3
        
        ll = ll - sum(sum(squeeze(phi(:,:,p,q)).*log(squeeze(phi(:,:,p,q)))));              % line 6
        
    end
end

end
% interestingly this function doenst take previous value of phi from outer loops into
% account
function phi_new = variationalUpdatesPhi(Y_pq,phi_pq,B,gamma_p,gamma_q,alpha,inner_iter)
%N = size(Y,1);
K = size(alpha,2);

deriv_phi_p = (psi(gamma_p) - repmat(psi(sum(gamma_p)),1,K))';
deriv_phi_q = (psi(gamma_q) - repmat(psi(sum(gamma_q)),1,K));
phi_new = exp( Y_pq.*log(B) + (1-Y_pq).*log(1-B) + repmat(deriv_phi_p,1,K) + repmat(deriv_phi_q, K, 1));
phi_new = phi_new./repmat(sum(sum(phi_new)),K,K);
end

function B = updateB(phi,Y,K)
N = size(Y,1);
B = zeros(K,K);
denBmat = zeros(K,K);

for p=1:N
    for q=1:N
        
        B = B + Y(p,q).*squeeze(phi(:,:,p,q));
        denBmat = denBmat + squeeze(phi(:,:,p,q));
    end
end
B = B./denBmat;
end

function gamma_p = updateGamma(phi,alpha,p,N)
gamma_p=alpha;

% I am not sure how squeeze function works in case of 4D array

for q=1:N
    gamma_p = gamma_p + sum(squeeze(phi(:,:,p,q)),2)' + sum(squeeze(phi(:,:,q,p)),1);
end

end


function alpha_new = updateAlpha(gamma,alpha)
N = size(gamma,1);
K = size(alpha,2);
% alpha
% try 
g = N.*(psi(sum(alpha)) - psi(alpha)) + sum(gamma,1) - sum(psi(sum(gamma,2))); % size 1*K
H = N.*(diag(psi(1,alpha)) - psi(1,sum(alpha)));
alpha_new = alpha+g*inv(H)*1e-4;
% catch exception
%     pi = gamma./repmat(sum(gamma,2),1,K)
% end
end

% function ll = getLogLikelihood()
% ll=0;
% 
% end


