function [alpha, B, pi, eta] = MMSB_sampling_poisson_tensor(Y,K,sampleIters, outer, L, T, filename)
% prior for B
%eta = ones(2,1)*0.1; % uniform prior
% prior for pi

eta = 2;	% eta is scalar for poisson
kappa = 3;
%eta=theta;
alpha = ones(K,1)*0.5; % uniform prior

num_users = size(Y,1);

threshold=1e-4;

% Block Matrix count
Nkk=zeros(K,K); % it emulates gamma distribution? cluster><cluster
Sum_kk=zeros(K,K,T); % needed by poisson distribution sampling, T for the tensor dimension

pi = zeros(num_users,K);

B = zeros(K,K,T);

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
        y_pq=Y(user_p,user_q,:);  %point to note: y is no more 0,1, we need 1,2
        Nkk(p_k,q_k)=Nkk(p_k,q_k)+1; % increment Ngh, y_pq not needed any more
        Sum_kk(p_k,q_k,:)=Sum_kk(p_k,q_k,:)+y_pq; % needed for poisson
        Nuk(user_p,p_k)=Nuk(user_p,p_k)+1;
        Nuk(user_q,q_k)=Nuk(user_q,q_k)+1;
        
    end
end

sum(sum(Sum_kk))

Nkk
Nuk

% Sum_kk
% Zuu
% return 
% TO BE NOTED send sampling probabilities only in the order they are expected.

%Gibbs Sampling 
% At present going for fixed number of iterations

for inner_iter=1:sampleIters
    inner_iter
    [Nkk,Nuk,Zuu,Sum_kk] = sampler(Nkk,Nuk,Zuu,eta,alpha,num_users,Y,K,Sum_kk,kappa,T);
%     Zuu
%     [x1_4,y1_4]=find(Zuu(:,:,1)==4);
% %     [x2_2,y2_2]=find(Zuu(:,:,2)==2);
%     ans_4 = Zuu(x1_4,y1_4,1);
%     [x2_4, y2_4] = find(ans_4==2);
%     x_4 = [x1_4,y1_4];
%     [x_c,y_c] = 
%     x_2 = [x2_2,y2_2];
%     diff=sum(abs(x_4-x_2),2);
%     idx = find(diff==0);
%     x1=x1_4(idx);
%     y1=y1_4(idx);
%     x1
%     y1
    ll(inner_iter)=calculate_joint_log_likelihood(Nkk,Nuk,K,num_users,alpha,eta,kappa,Sum_kk,Zuu);
%     Nkk
%     Nuk
    ll(inner_iter)
%     sum(sum(Sum_kk))
    if inner_iter>1 && abs(ll(inner_iter)-ll(inner_iter-1))<threshold
        break;
    end
end

temp_pi=zeros(num_users,K);
for iter=1:outer
    for inner=1:L
        [Nkk,Nuk,Zuu,Sum_kk] = sampler(Nkk,Nuk,Zuu,eta,alpha,num_users,Y,K,Sum_kk,kappa,T);
    end
    iter
    ll(sampleIters+iter)=calculate_joint_log_likelihood(Nkk,Nuk,K,num_users,alpha,eta,kappa,Sum_kk,Zuu);
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
            B(g,h,:)= B(g,h,:) + (Sum_kk(g,h,:)+kappa*ones(1,1,T))/(Nkk(g,h) + 1/eta);
        end
    end
    
end
pi=pi/outer;
B=B/outer;

save(strcat(filename,'_poisson.mat'),'pi');

end

function [ll]=calculate_joint_log_likelihood(Nkk,Nuk,K,num_users,alpha,eta,kappa,Sum_kk,Zuu)
ll=0;
% Nkk

for g=1:K
%     [factorial_x,factorial_y] = find(Zuu)
    for h=1:K        
        
        ll=ll +gammaln(Sum_kk(g,h,1)+kappa) -(Sum_kk(g,h,1)+kappa)*(Nkk(g,h)+1/eta)-kappa*log(eta)-gammaln(kappa); %- sumSigma gammaln(Y_pq+1)    %temp for tensor
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

function [Nkk,Nuk,Zuu,Sum_kk] = sampler(Nkk,Nuk,Zuu,eta,alpha,num_users,Y,K,Sum_kk,kappa,T)
for user_p=1:num_users
    for user_q=1:num_users
        if user_p==user_q       %if not then we will be sampling p->q twice. Wrong : y_pq != Y_qp
            continue;
        end
        %get current assignment of k to p->q and q->p
        p_k = Zuu(user_p,user_q,1);
        q_k= Zuu(user_q,user_p,2);
        
        y_pq=Y(user_p,user_q,:);  %point to note: y is 0,1, we need 1,2
        multi_probs=calculate_Mult_Probs(Nkk,Nuk(user_p,:),Nuk(user_q,:),K,y_pq,eta,alpha,Sum_kk,kappa,p_k,q_k,T);
        
%         multi_probs
        
        %decrement current counts
        Nkk(p_k,q_k)=Nkk(p_k,q_k)-1; % decrement Ngh, y_pq not needed for poisson
        Sum_kk(p_k,q_k,:) = Sum_kk(p_k,q_k,:)-y_pq;
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
        Sum_kk(p_k,q_k,:) = Sum_kk(p_k,q_k,:)+y_pq;
        Nuk(user_p,p_k)=Nuk(user_p,p_k)+1;
        Nuk(user_q,q_k)=Nuk(user_q,q_k)+1;
%         sum(sum(Sum_kk))
%         Sum_kk
    end
end
end

function [multi_probs]=calculate_Mult_Probs(Nkk,Npk,Nqk,K,y_pq,eta,alpha,Sum_kk,kappa,curr_p,curr_q,T)
multi_probs=zeros(K*K,1);
indx=0;
max=0;
for p_k=1:K
    for q_k=1:K
        indx=indx+1;
        n_kk = Nkk(p_k,q_k);
        alpha_p=alpha(p_k);
        alpha_q=alpha(q_k);
        
        sum_local = Sum_kk(p_k,q_k,:);
%         sum_local
        if p_k==curr_p
            alpha_p = alpha_p-1;
        end
        if q_k==curr_q
            alpha_q = alpha_q-1;      
        end
        if p_k==curr_p && q_k==curr_q
            n_kk = n_kk-1;
            sum_local = sum_local - y_pq;
        end
        p=1/((1/eta)+n_kk+1);
%         size(sum_local)
%         sum_local
%         sum(sum_local)
%         reshape(sum_local,T,1);
        sum_local = sum_local + ones(1,1,T)*kappa;
%         sum_local
%         p
%         y_pq
        block_partVec = log(nbinpdf(y_pq,sum_local,ones(1,1,T)*p));
        block_part = sum(block_partVec);
%         block_part
        log_probs = block_part + log(((Npk(p_k)+alpha_p))) + log(((Nqk(q_k)+alpha_q)));
        probs = exp(log_probs);
        multi_probs(indx)= exp(probs);
%         indx
%         probs
    end
end
% sum(sum(Sum_kk))
% sum_y
% y_pq
% multi_probs = multi_probs/max;
% multi_probs     
end

function [k_p,k_q] = multinomial_bivariate_sample(mult_probs,K)
%     mult_probs
    mult_probs = mult_probs/sum(mult_probs);
    mult_probs = cumsum(mult_probs);
    index=find(mult_probs>rand,1);
    k_p=ceil(index/K);
%     k_p
    k_q=mod(index,K);
    if k_q==0
        k_q=K;
    end    
%     mult_probs
end