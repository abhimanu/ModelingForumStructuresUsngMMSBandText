function [alpha, B, pi, eta] = MMSB_sampling_poisson(Y,K,sampleIters, outer, L)
% prior for B
%eta = ones(2,1)*0.1; % uniform prior
% prior for pi

eta = 2;	% eta is scalar for poisson
kappa = 3;
%eta=theta;
alpha = ones(K,1)*0.5; % uniform prior

num_users = size(Y,1);

threshold=1e-2;

% Block Matrix count
Nkk=zeros(K,K); % it emulates gamma distribution? cluster><cluster
Sum_kk=zeros(K,K); % needed by poisson distribution sampling

pi = zeros(num_users,K);

B = zeros(K,K);

Zuu = zeros(num_users,num_users,2); % user><user stores index of cluster, z_pq; two sets to distinguish between z_p->q and z_q<-p
Nuk = zeros(num_users, K);  %user><cluster, n_user_k
ll=zeros(sampleIters+outer*L,1);
%intialization 
for user_p=1:num_users
    for user_q=1:num_users
        if user_p==user_q
            continue;
        end
        % sample Z_pq and Z_qp togethr
        [p_k,q_k]=multinomial_bivariate_sample(ones(K*K,1)/(K*K),K);
        Zuu(user_p,user_q,1)=p_k;
        Zuu(user_q,user_p,2)=q_k;
        y_pq=Y(user_p,user_q);  %point to note: y is no more 0,1, we need 1,2
        Nkk(p_k,q_k)=Nkk(p_k,q_k)+1; % increment Ngh, y_pq not needed any more
        Sum_kk(p_k,q_k)=Sum_kk(p_k,q_k)+y_pq; % needed for poisson
        Nuk(user_p,p_k)=Nuk(user_p,p_k)+1;
        Nuk(user_q,q_k)=Nuk(user_q,q_k)+1;
    end
end

sum(sum(Sum_kk))

Nkk
Nuk

% return 
% TO BE NOTED send sampling probabilities only in the order they are expected.

%Gibbs Sampling 
% At present going for fixed number of iterations

for inner_iter=1:sampleIters
    inner_iter
    [Nkk,Nuk,Zuu] = sampler(Nkk,Nuk,Zuu,eta,alpha,num_users,Y,K,Sum_kk,kappa);
    ll(inner_iter)=calculate_joint_log_likelihood(Nkk,Nuk,K,num_users,alpha,eta,kappa,Sum_kk);
%     Nkk
%     Nuk
    ll(inner_iter)
    if inner_iter>1 && abs(ll(inner_iter)-ll(inner_iter-1))<threshold
        break;
    end
end

temp_pi=zeros(num_users,K);
for iter=1:outer
    for inner=1:L
        [Nkk,Nuk,Zuu] = sampler(Nkk,Nuk,Zuu,eta,alpha,num_users,Y,K,Sum_kk,kappa);
    end
    iter
    ll(sampleIters+iter)=calculate_joint_log_likelihood(Nkk,Nuk,K,num_users,alpha,eta,kappa,Sum_kk);
    ll(sampleIters+iter)
    %estimate Pi
    for u=1:num_users
        for k=1:K
            temp_pi(u,k)=Nuk(u,k)+alpha(k);
        end
    end
    
    temp_pi=temp_pi./repmat(sum(temp_pi,2),1,K);
    
    pi=pi+temp_pi;
    
    %estimate B, block matrix
    for g=1:K
        for h =1:K
            B(g,h)= B(g,h) + (Sum_kk(g,h)+kappa)/(Nkk(g,h) + 1/eta);
        end
    end
    
end
pi=pi/outer;
B=B/outer;

save(strcat(filename,'_poisson.mat'),'pi');

end

function [ll]=calculate_joint_log_likelihood(Nkk,Nuk,K,num_users,alpha,eta,kappa,Sum_kk)
ll=0;
% Nkk

for g=1:K
    for h=1:K
        ll=ll-(Sum_kk(g,h)+kappa)*(Nkk(g,h)+1/eta)-kappa*log(eta)-gammaln(kappa);       %TODO    +gammaln(Sum_kk(g,h)+kappa)-sumSigma gammaln(Y_pq+1)    
    end
end

for p=1:num_users
    for k=1:K
        ll=ll+gammaln(Nuk(p,k)+alpha(k))-gammaln(alpha(k));
%         ll
    end
    ll=ll+gammaln(sum(alpha))-gammaln(sum(Nuk(p,:))+sum(alpha));
end 
end

function [Nkk,Nuk,Zuu] = sampler(Nkk,Nuk,Zuu,eta,alpha,num_users,Y,K,Sum_kk,kappa)
for user_p=1:num_users
    for user_q=1:num_users
        if user_p==user_q       %if not then we will be sampling p->q twice. Wrong : y_pq != Y_qp
            continue;
        end
        y_pq=Y(user_p,user_q);  %point to note: y is 0,1, we need 1,2
        multi_probs=calculate_Mult_Probs(Nkk,Nuk(user_p,:),Nuk(user_q,:),K,y_pq,eta,alpha,Sum_kk,kappa);
        
%         multi_probs
        %get current assignment of k to p->q and q->p
        p_k = Zuu(user_p,user_q,1);
        q_k= Zuu(user_q,user_p,2);
        %decrement current counts
        Nkk(p_k,q_k)=Nkk(p_k,q_k)-1; % decrement Ngh, y_pq not needed for poisson
        Sum_kk(p_k,q_k) = Sum_kk(p_k,q_k)-y_pq;
        Nuk(user_p,p_k)=Nuk(user_p,p_k)-1;
        Nuk(user_q,q_k)=Nuk(user_q,q_k)-1;
        
        % sample Z_pq and Z_qp together
        [p_k,q_k]=multinomial_bivariate_sample(multi_probs,K);
%         p_k
        Zuu(user_p,user_q,1)=p_k;
        Zuu(user_q,user_p,2)=q_k;
        %increment the current counts
%         y_pq=Y(user_p,user_q);          %  remeber to get the new y_pq
        Nkk(p_k,q_k)=Nkk(p_k,q_k)+1; % increment Ngh, y_pq not needed for poisson
        Sum_kk(p_k,q_k) = Sum_kk(p_k,q_k)+y_pq;
        Nuk(user_p,p_k)=Nuk(user_p,p_k)+1;
        Nuk(user_q,q_k)=Nuk(user_q,q_k)+1;
    end
end
end

function [multi_probs]=calculate_Mult_Probs(Nkk,Npk,Nqk,K,y_pq,eta,alpha,Sum_kk,kappa)
multi_probs=zeros(K*K,1);
indx=0;
max=0;
for p_k=1:K
    for q_k=1:K
        indx=indx+1;
        log_term = 0;
        sum_y=Sum_kk(p_k,q_k)+kappa;
        % y_pq is different than p_k and q_k
        if y_pq==0
            y_pq=1;
            sum_y=sum_y+1;
        end
        for i=1:y_pq
            log_term=log_term+log(sum_y-i);
        end
        log_term
%         if log_term>1e9
%             
%         end
        
        k = (sum_y-y_pq)*log(Nkk(p_k,q_k)-1 + (1/eta)) - sum_y*log(Nkk(p_k,q_k) + (1/eta)) - log(y_pq) ;
        k
        sum_y
        y_pq
        log_term = log_term +k;%- sum_y*log(Nkk(p_k,q_k) + (1/eta)) - log(y_pq) + (sum_y-y_pq)*log(Nkk(p_k,q_k)-1 + (1/eta));   %term for poisson distribution
%         sum_y*log(Nkk(p_k,q_k) + (1/eta)) - (sum_y-y_pq)*log(Nkk(p_k,q_k)-1 + (1/eta))
%         log_term
%         term_exp = exp(log_term);
%         term_exp
        multi_probs(indx)= log_term + log((Npk(p_k)+alpha(p_k)-1))-log(sum(Npk)+alpha(p_k)-1) + log(Nqk(q_k)+alpha(q_k)-1) - log(sum(Nqk)+alpha(q_k)-1);
        mprob=multi_probs(indx);
        mprob
        if indx==1
            max=multi_probs(indx);
        end
        if max<multi_probs(indx)
            max=multi_probs(indx);
        end
    end
end
sum(sum(Sum_kk))
% sum_y
% y_pq
multi_probs = multi_probs/max;
% multi_probs     
end

function [k_p,k_q] = multinomial_bivariate_sample(mult_probs,K)
    mult_probs = mult_probs/sum(mult_probs);
    mult_probs = cumsum(mult_probs);
    index=find(mult_probs>rand,1);
    k_p=ceil(index/K);
%     k_p
    k_q=mod(index,K);
    if k_q==0
        k_q=K;
    end    
end
