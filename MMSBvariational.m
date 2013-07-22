function[pi,B,alpha] = MMSBvariational(Y, K, maxit, inner_iter)
N = size(Y,1);
% gamma = repmat(ones(1,K)*2*N/K, N, 1);

alpha = 0.5+(rand(1,K)-0.5)*0.1
B = eye(K).*0.99+1e-6;
log(B)
log(1-B)
%rB = rand(K,K)
% gamma = repmat(alpha,N,1)+(rand(N,K)-0.5).*0.1
% gamma = abs((rand(N,K)-0.5)*N)
gamma = initializeGamma(N,K)
% gamma(gamma<=0)=0.1;
% gamma
ll_iters = [];

for iter=1:maxit
    [gamma,B,ll, alpha] = updateGammasAndBlockMat(gamma,B,alpha,Y,inner_iter);
    
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

function [gamma, B, ll, alpha] = updateGammasAndBlockMat(gamma,B,alpha,Y,inner_iter)
N = size(Y,1);
K = size(alpha,2);
phi = ones(N,N,2,K)*1.0/K;

ll=0;
% ll = ll + N*(gammaln(sum(alpha)) - sum(gammaln(alpha)));                    % line 4
for p=1:N
%     deriv_phi_p = psi(gamma(p,:)) - repmat(psi(sum(gamma(p,:))),1,K);
%     ll = ll + sum((alpha-1).*deriv_phi_p);                                  % line 4
    
%     ll = ll - gammaln(sum(gamma(p,:))) + sum(gammaln(gamma(p,:)));          % line 5
%     ll = ll - sum((gamma(p,:)-1).*deriv_phi_p);                             % line 5
    
%     gamma_p = alpha;
    for q=1:N
        % calculate log-likelihoods
%         f_B = Y(p,q)*log(B) + (1-Y(p,q))*log(1-B);
%         ll = ll + sum(sum((squeeze(phi(p,q,1,:))*squeeze(phi(p,q,2,:))') .* f_B));          % line 1
%         ll = ll + sum(squeeze(phi(p,q,1,:))'.*deriv_phi_p);                           % line 2
        
%         deriv_phi_q = psi(gamma(q,:)) - repmat(psi(sum(gamma(q,:))),1,K);
%         ll = ll + sum(squeeze(phi(p,q,2,:))'.*deriv_phi_q);                           % line 3
        
%         ll = ll - sum(squeeze(phi(p,q,1,:)).*log(squeeze(phi(p,q,1,:))));                     % line 6
%         ll = ll - sum(squeeze(phi(p,q,2,:)).*log(squeeze(phi(p,q,2,:))));                     % line 6
        
%         old_phi1 = phi(p,q,1,:);
%         old_phi2 = phi(p,q,2,:);
%         old_alpha = alpha;
        
        % get new phi_pq
%        temp_phi1 = variationalUpdatesPhi(Y(p,q),phi(2,p,q),B,1);
        phi(p,q,:,:) = variationalUpdatesPhi(Y(p,q),phi(p,q),B,gamma(p,:),gamma(q,:),alpha,inner_iter);
%        phi(1,p,q) = temp_phi1;

        % update gamma
%         squeeze(phi(p,q,1,:))'
%         squeeze(phi(p,q,2,:))'

%          gamma_p = gamma_p + squeeze(phi(p,q,1,:))' + squeeze(phi(p,q,2,:))';
        gamma(p,:) = updateGamma(phi,alpha,p,N);

        % update B
         B=updateB(phi,Y,K);              % write a more efficient code for updates
%          alpha = updateAlpha(gamma,alpha);
        
%         gamma_p = gamma(p,:) - old_alpha + alpha - squeeze(old_phi1)' - squeeze(old_phi2)'...
%             + squeeze(phi(p,q,1,:))' + squeeze(phi(p,q,2,:))';
%         gamma_p(gamma_p<=0) = 0.1;
%         gamma(p,:) = gamma_p;
%         getLogLikelihood(gamma,B,alpha,Y,phi)
    end
%     gamma(p,:) = gamma_p;
%     B=updateB(phi,Y,K);              % write a more efficient code for updates

end
%ll
     alpha = updateAlpha(gamma,alpha);
     ll = getLogLikelihood(gamma,B,alpha,Y,phi)
     pi = gamma./repmat(sum(gamma,2),1,K)
end

function ll = getLogLikelihood(gamma,B,alpha,Y,phi)
N = size(Y,1);
K = size(alpha,2);
ll=0;
ll = ll + N*(gammaln(sum(alpha)) - sum(gammaln(alpha)));                    % line 4
for p=1:N
    deriv_phi_p = psi(gamma(p,:)) - repmat(psi(sum(gamma(p,:))),1,K);
    ll = ll + sum((alpha-1).*deriv_phi_p);                                  % line 4
    
    ll = ll - gammaln(sum(gamma(p,:))) + sum(gammaln(gamma(p,:)));          % line 5
    ll = ll - sum((gamma(p,:)-1).*deriv_phi_p);                             % line 5

    for q=1:N
        % calculate log-likelihoods
        f_B = Y(p,q)*log(B) + (1-Y(p,q))*log(1-B);
        ll = ll + sum(sum((squeeze(phi(p,q,1,:))*squeeze(phi(p,q,2,:))') .* f_B));          % line 1
        ll = ll + sum(squeeze(phi(p,q,1,:))'.*deriv_phi_p);                           % line 2
        
        deriv_phi_q = psi(gamma(q,:)) - repmat(psi(sum(gamma(q,:))),1,K);
        ll = ll + sum(squeeze(phi(p,q,2,:))'.*deriv_phi_q);                           % line 3
        
        ll = ll - sum(squeeze(phi(p,q,1,:)).*log(squeeze(phi(p,q,1,:))));                     % line 6
        ll = ll - sum(squeeze(phi(p,q,2,:)).*log(squeeze(phi(p,q,2,:))));                     % line 6

    end
end

end
% interestingly this function doenst take previous value of phi from outer loops into
% account
function phi_new = variationalUpdatesPhi(Y_pq,phi_pq,B,gamma_p,gamma_q,alpha,inner_iter)
%N = size(Y,1);
K = size(alpha,2);
phi_1 = ones(1,K)*1.0/K;
phi_2 = ones(1,K)*1.0/K;
phi_temp = ones(1,K)*1.0/K;
iter=0;
gamma_p;
gamma_q;

while iter<inner_iter
    iter = iter+1;
    for g=1:K
        psi_factor = psi(gamma_p(g))-psi(sum(gamma_p));
        b_factor = sum(phi_2.*(Y_pq*log(B(g,:))+(1-Y_pq).*(log(1-B(g,:)))));
        phi_temp(g) = exp(psi_factor+b_factor);
    end
    phi_temp = phi_temp./sum(phi_temp);
    for h=1:K
        psi_factor = psi(gamma_q(h))-psi(sum(gamma_q));
        b_factor = phi_1*(Y_pq*log(B(:,h))+(1-Y_pq).*(log(1-B(:,h))));
        phi_2(h) = exp(psi_factor+b_factor);
    end
    phi_2 = phi_2./sum(phi_2);
    phi_1 = phi_temp;
end
phi_1;
phi_2;
phi_new = [phi_1; phi_2];
end

function B = updateB(phi,Y,K)
N = size(Y,1);
B = zeros(K,K);
denB = 0;
for p=1:N
    for q=1:N
        denBmat = squeeze(phi(p,q,1,:))*squeeze(phi(p,q,2,:))';
        B = B + Y(p,q)*denBmat;
        denB = denB + sum(sum(denBmat));
    end
end
B = B/denB;
end

function gamma_p = updateGamma(phi,alpha,p,N)
gamma_p=alpha;
for q=1:N
    gamma_p = gamma_p + squeeze(phi(p,q,1,:))' + squeeze(phi(p,q,2,:))';
end
end


function alpha_new = updateAlpha(gamma,alpha)
N = size(gamma,1);
K = size(alpha,2);
% alpha
% try 
g = N.*(psi(sum(alpha)) - psi(alpha)) + sum(gamma,1) - sum(psi(sum(gamma,2))); % size 1*K
H = N.*(diag(psi(1,alpha)) - psi(1,sum(alpha)));
alpha_new = alpha+g*inv(H)*1e-5;
% catch exception
%     pi = gamma./repmat(sum(gamma,2),1,K)
% end
end

% function ll = getLogLikelihood()
% ll=0;
% 
% end
