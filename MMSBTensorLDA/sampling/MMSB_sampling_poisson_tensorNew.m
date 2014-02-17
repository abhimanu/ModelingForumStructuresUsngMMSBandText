function [alpha, B, B2, pi, eta] = MMSB_sampling_poisson_tensorNew(Y1,Y2,K,sampleIters, outer, L, filename)
% prior for B
%eta = ones(2,1)*0.1; % uniform prior
% prior for pi

eta = 10;	% eta is scalar for poisson
kappa = 1;
%eta=theta;
alpha = ones(K,1)*0.1; % uniform prior

num_users = size(Y1,1);

threshold=1e-4;

% Block Matrix count
Nkk=zeros(K,K); % it emulates gamma distribution? cluster><cluster
Sum_kk1=zeros(K,K); % needed by poisson distribution sampling
Sum_kk2=zeros(K,K); % needed by poisson distribution sampling

pi = zeros(num_users,K);

B = zeros(K,K);

Zuu = zeros(num_users,num_users,2); % user><user stores index of cluster, z_pq; two sets to distinguish between z_p->q and z_q<-p
Nuk = zeros(num_users, K);  %user><cluster, n_user_k
ll=zeros(sampleIters+outer,1);
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
        y_pq1=Y1(user_p,user_q);  %point to note: y is no more 0,1, we need 1,2
        y_pq2=Y2(user_p,user_q);
        Nkk(p_k,q_k)=Nkk(p_k,q_k)+1; % increment Ngh, y_pq not needed any more
        
        Sum_kk1(p_k,q_k)=Sum_kk1(p_k,q_k)+y_pq1; % needed for poisson
        Sum_kk2(p_k,q_k)=Sum_kk2(p_k,q_k)+y_pq2; 
        
        Nuk(user_p,p_k)=Nuk(user_p,p_k)+1;
        Nuk(user_q,q_k)=Nuk(user_q,q_k)+1;
        
    end
end

sum(sum(Sum_kk1))
sum(sum(Sum_kk2))

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
    [Nkk,Nuk,Zuu,Sum_kk1, Sum_kk2] = sampler(Nkk,Nuk,Zuu,eta,alpha,num_users,Y1,Y2,K,Sum_kk1,Sum_kk2,kappa);
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
    ll(inner_iter)=calculate_joint_log_likelihood(Nkk,Nuk,K,num_users,alpha,eta,kappa,Sum_kk1,Sum_kk2,Zuu,Y1,Y2);
%     Nkk
%     Nuk
    ll(inner_iter)
%     sum(sum(Sum_kk))
%     if inner_iter>1 && abs(ll(inner_iter)-ll(inner_iter-1))<threshold
%         break;
%     end
end

temp_pi=zeros(num_users,K);
for iter=1:outer
    for inner=1:L
        [Nkk,Nuk,Zuu,Sum_kk1,Sum_kk2] = sampler(Nkk,Nuk,Zuu,eta,alpha,num_users,Y1,Y2,K,Sum_kk1,Sum_kk2,kappa);
    end
    iter
    inner_iter = inner_iter+1;
    ll(inner_iter)=calculate_joint_log_likelihood(Nkk,Nuk,K,num_users,alpha,eta,kappa,Sum_kk1,Sum_kk2,Zuu,Y1,Y2);
    ll(inner_iter)
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
            B(g,h)= B(g,h) + (Sum_kk1(g,h)+kappa)/(Nkk(g,h) + 1/eta);
        end
    end
    
end
pi=pi/outer;
B=B/outer;
B2=B; %dont care

save(strcat(filename,'_PI.mat'),'pi');

% size(ll)
inner_iter
plot(1:inner_iter,ll);
saveas(gcf,strcat(filename,'_LL.png'));

end

function [ll]=calculate_joint_log_likelihood(Nkk,Nuk,K,num_users,alpha,eta,kappa,Sum_kk1,Sum_kk2,Zuu,Y1,Y2)
ll=0;
% Nkk

for g=1:K
%     [factorial_x,factorial_y] = find(Zuu)
    for h=1:K        
        [sumLnGammaY_pq1, sum_gh1] = getSumLnGammaY_pq(Zuu, g, h, Y1);
        [sumLnGammaY_pq2, sum_gh2] = getSumLnGammaY_pq(Zuu, g, h, Y2);
%         Sum_kk(g,h)
        if sum_gh1~=Sum_kk1(g,h)
            'screwed badly'
            return;
        end
        ll=ll +gammaln(Sum_kk1(g,h)+kappa) -(Sum_kk1(g,h)+kappa)*(Nkk(g,h)+1/eta)-kappa*log(eta)-gammaln(kappa); %- sumSigma gammaln(Y_pq+1)    
        ll = ll-sumLnGammaY_pq1;
        
        ll = ll + gammaln(Sum_kk2(g,h)+kappa) -(Sum_kk2(g,h)+kappa)*(Nkk(g,h)+1/eta);
        ll = ll-sumLnGammaY_pq2;
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

function [sumLnGammaY_pq,sum_gh] = getSumLnGammaY_pq(Zuu,g,h,Y)
%     sumLnGammaY_pq = 0;
    Z1 = Zuu(:,:,1);
    Z2 = Zuu(:,:,2);
    [r1,c1] = find(Z1==g);
    Z2_2 = Z2(sub2ind(size(Z2),c1,r1)); % p->q become q->p
    r = r1(Z2_2==h);
    c = c1(Z2_2==h);
    y_gh = Y(sub2ind(size(Y),r,c));
    sum_gh=sum(y_gh);
    sumLnGammaY_pq = sum(gammaln(y_gh+1));    
end

function [Nkk,Nuk,Zuu,Sum_kk1,Sum_kk2] = sampler(Nkk,Nuk,Zuu,eta,alpha,num_users,Y1,Y2,K,Sum_kk1,Sum_kk2,kappa)
for user_p=1:num_users
    for user_q=1:num_users
        if user_p==user_q       %if not then we will be sampling p->q twice. Wrong : y_pq != Y_qp
            continue;
        end
        %get current assignment of k to p->q and q->p
        p_k = Zuu(user_p,user_q,1);
        q_k= Zuu(user_q,user_p,2);
        
        y_pq1=Y1(user_p,user_q);  %point to note: y is 0,1, we need 1,2
        y_pq2=Y2(user_p,user_q);
        multi_probs=calculate_Mult_Probs(Nkk,Nuk(user_p,:),Nuk(user_q,:),K,y_pq1,y_pq2,eta,alpha,Sum_kk1,Sum_kk2,kappa,p_k,q_k);
        
%         multi_probs
        
        %decrement current counts
        Nkk(p_k,q_k)=Nkk(p_k,q_k)-1; % decrement Ngh, y_pq not needed for poisson
        Sum_kk1(p_k,q_k) = Sum_kk1(p_k,q_k)-y_pq1;
        Sum_kk2(p_k,q_k) = Sum_kk2(p_k,q_k)-y_pq2;
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
        Sum_kk1(p_k,q_k) = Sum_kk1(p_k,q_k)+y_pq1;
        Sum_kk2(p_k,q_k) = Sum_kk2(p_k,q_k)+y_pq2;
        Nuk(user_p,p_k)=Nuk(user_p,p_k)+1;
        Nuk(user_q,q_k)=Nuk(user_q,q_k)+1;
%         sum(sum(Sum_kk))
%         Sum_kk
    end
end
end

function [multi_probs]=calculate_Mult_Probs(Nkk,Npk,Nqk,K,y_pq1,y_pq2,eta,alpha,Sum_kk1,Sum_kk2,kappa,curr_p,curr_q)
multi_probs=zeros(K*K,1);
indx=0;
max=0;
for p_k=1:K
    for q_k=1:K
        indx=indx+1;
        n_kk = Nkk(p_k,q_k);
        alpha_p=alpha(p_k);
        alpha_q=alpha(q_k);
        
        sum_local1 = Sum_kk1(p_k,q_k);
        sum_local2 = Sum_kk2(p_k,q_k);
        % changed here to only substract when when both p->q=g and p<-q=h
%         sum_local
%         if p_k==curr_p
%             alpha_p = alpha_p-1;
%         end
%         if q_k==curr_q
%             alpha_q = alpha_q-1;      
%         end
        % changed here to only substract when when both p->q=g and p<-q=h
        if p_k==curr_p && q_k==curr_q
            n_kk = n_kk-1;
            sum_local1 = sum_local1 - y_pq1;
            sum_local2 = sum_local2 - y_pq2;
            alpha_p = alpha_p-1;
            alpha_q = alpha_q-1;
        end
        p=1/((1/eta)+n_kk+1);
        p=1-p;
        sum_local1 = sum_local1 + kappa;
        sum_local2 = sum_local2 + kappa;
%         sum_local
%         p
%         y_pq
        block_part1 = log(nbinpdf(y_pq1,sum_local1,p));        % nbinpdf returns zero if y_pq is not and integer
        block_part2 = log(nbinpdf(y_pq2,sum_local2,p));
%         block_part;
        log_probs = block_part1 + block_part2 + log(((Npk(p_k)+alpha_p))) + log(((Nqk(q_k)+alpha_q)));
        probs = exp(log_probs);
        multi_probs(indx)= probs;
%         indx
%         log_probs
%         probs
    end
end
% multi_probs
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
%     mult_probs
%     
%     k_p
%     k_p
    k_q=mod(index,K);
    if k_q==0
        k_q=K;
    end    
%     mult_probs
end