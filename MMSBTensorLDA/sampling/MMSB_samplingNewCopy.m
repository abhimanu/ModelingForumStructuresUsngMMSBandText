function [alpha, B, pi, eta] = MMSB_samplingNewCopy(Y,K,sampleIters, outer, L, filename)
% prior for B
eta = ones(2,1)*0.1; % uniform prior
% prior for pi
alpha = ones(K,1)*0.1; % uniform prior

num_users = size(Y,1);

threshold=1e-4;

% Block Matrix count
Nkk=zeros(K,K,2);       %tensor since it emulates Beta distribution, cluster><cluster

pi = zeros(num_users,K);

B = zeros(K,K);

Zuu = zeros(num_users,num_users,2,2); % user><user stores index of cluster, z_pq; two sets to distinguish between z_p->q and z_q<-p
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
        
        y_pq=Y(user_p,user_q);  %point to note: y is 0,1, we need 1,2
        Zuu(user_p,user_q,y_pq+1,1)=p_k;
        Zuu(user_q,user_p,y_pq+1,2)=q_k;
        Nkk(p_k,q_k,y_pq+1)=Nkk(p_k,q_k,y_pq+1)+1; % increment Ngh,y_pq
        Nuk(user_p,p_k)=Nuk(user_p,p_k)+1;
        Nuk(user_q,q_k)=Nuk(user_q,q_k)+1;
    end
end

synccKK(Zuu,K)

Nkk
Nuk

% return 
% TO BE NOTED send sampling probabilities only in the order they are expected.

%Gibbs Sampling 
% At present going for fixed number of iterations
inner_iter=1;
for inner_iter=1:sampleIters
    inner_iter
    [Nkk,Nuk,Zuu] = sampler(Nkk,Nuk,Zuu,eta,alpha,num_users,Y,K);
    ll(inner_iter)=calculate_joint_log_likelihood(Nkk,Nuk,K,num_users,alpha,eta);
%     Nkk
%     Nuk
    ll(inner_iter)
    synccKK(Zuu,K)
%     if inner_iter>1 && abs(ll(inner_iter)-ll(inner_iter-1))<threshold
%         break;
%     end
end

temp_pi=zeros(num_users,K);
for iter=1:outer
    for inner=1:L
        [Nkk,Nuk,Zuu] = sampler(Nkk,Nuk,Zuu,eta,alpha,num_users,Y,K);
    end
    iter
    synccKK(Zuu,K)
    
    inner_iter = inner_iter+1;
    ll(inner_iter)=calculate_joint_log_likelihood(Nkk,Nuk,K,num_users,alpha,eta);
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
            B(g,h)= B(g,h) + (Nkk(g,h,2)+eta(2))/(sum(Nkk(g,h))+sum(eta));
        end
    end
    
end
pi=pi/outer;
B=B/outer;

save(strcat(filename,'_PI.mat'),'pi');

% size(ll)
inner_iter
plot(1:inner_iter,ll);
saveas(gcf,strcat(filename,'_LL.png'));

end

function [ll]=calculate_joint_log_likelihood(Nkk,Nuk,K,num_users,alpha,eta)
ll=0;
% Nkk

for g=1:K
    for h=1:K
        for y=1:2
            ll=ll+gammaln(Nkk(g,h,y)+eta(y))-gammaln(eta(y));
%             ll
        end
        ll=ll+gammaln(sum(eta))-gammaln(sum(Nkk(g,h,:))+sum(eta));
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

function [] = synccKK(Zuu,K)
%     sumLnGammaY_pq = 0;
    Nkk_sync=zeros(K,K,2);
    Z_1 = Zuu(:,:,:,1);
    Z_2 = Zuu(:,:,:,2);
    for y_pq=1:2
        Z1 = Z_1(:,:,y_pq);
        Z2 = Z_2(:,:,y_pq);
        for g = 1:K
            for h=1:K
                [r1,c1] = find(Z1==g);
                Z2_2 = Z2(sub2ind(size(Z2),c1,r1)); % p->q become q->p
                r = r1(Z2_2==h);
%               c = c1(Z2_2==h);
                Nkk_sync(g,h,y_pq) = size(r,1);
            end
        end
    end
    sum_sync=sum(sum(sum(Nkk_sync)))
end


function [Nkk,Nuk,Zuu] = sampler(Nkk,Nuk,Zuu,eta,alpha,num_users,Y,K)
for user_p=1:num_users
    for user_q=1:num_users
        if user_p==user_q       %if not then we will be sampling p->q twice. Wrong : y_pq != Y_qp
            continue;
        end
        %get current assignment of k to p->q and q->p
        
        
        y_pq=Y(user_p,user_q);  %point to note: y is 0,1, we need 1,2
        p_k = Zuu(user_p,user_q,y_pq+1,1);
        q_k= Zuu(user_q,user_p,y_pq+1,2);
        
        multi_probs=calculate_Mult_Probs(Nkk,Nuk(user_p,:),Nuk(user_q,:),K,y_pq+1,eta,alpha,p_k,q_k);
        
        %decrement current counts
        Nkk(p_k,q_k,y_pq+1)=Nkk(p_k,q_k,y_pq+1)-1; % decrement Ngh,y_pq
        Nuk(user_p,p_k)=Nuk(user_p,p_k)-1;
        Nuk(user_q,q_k)=Nuk(user_q,q_k)-1;
        
        % sample Z_pq and Z_qp together
        [p_k,q_k]=multinomial_bivariate_sample(multi_probs,K);
        
        Zuu(user_p,user_q,y_pq+1,1)=p_k;
        Zuu(user_q,user_p,y_pq+1,2)=q_k;
        Nkk(p_k,q_k,y_pq+1)=Nkk(p_k,q_k,y_pq+1)+1; % increment Ngh,y_pq
        Nuk(user_p,p_k)=Nuk(user_p,p_k)+1;
        Nuk(user_q,q_k)=Nuk(user_q,q_k)+1;
    end
end
end

function [multi_probs]=calculate_Mult_Probs(Nkk,Npk,Nqk,K,y_pq,eta,alpha,curr_p,curr_q)
multi_probs=zeros(K*K,1);
indx=0;
sum_eta=sum(eta);
for p_k=1:K
    for q_k=1:K
        indx=indx+1;
        alpha_p=alpha(p_k);
        alpha_q=alpha(q_k);
        eta_local=eta(y_pq);
%         if p_k==curr_p
%             alpha_p = alpha_p-1;
%         end
%         if q_k==curr_q
%             alpha_q = alpha_q-1;      
%         end
        if p_k==curr_p && q_k==curr_q
            eta_local = eta_local-1;
            sum_eta = sum_eta-1;
            alpha_p = alpha_p-1;
            alpha_q = alpha_q-1;
        end
        multi_probs(indx)=((Nkk(p_k,q_k,y_pq)+eta_local)/(sum(Nkk(p_k,q_k))+sum_eta)) * ((Npk(p_k)+alpha_p)) * ((Nqk(q_k)+alpha_q));
    end
end
% multi_probs
end

function [k_p,k_q] = multinomial_bivariate_sample(mult_probs,K)
    mult_probs = mult_probs/sum(mult_probs);
    mult_probs = cumsum(mult_probs);
    index=find(mult_probs>rand,1);
    k_p=ceil(index/K);
    k_q=mod(index,K);
    if k_q==0
        k_q=K;
    end    
end