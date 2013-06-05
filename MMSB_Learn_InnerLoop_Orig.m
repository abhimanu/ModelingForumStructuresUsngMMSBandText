function [gammahat, B, lastll] = MMSB_Learn_InnerLoop_Orig(y, alpha, B, gammahat, maxit)
% y: N*N
% alpha: 1*K
% B: K*K
% gammahat: N*K

N=size(y,1);
K=size(B,1);

lastll = 1; % invalid value, ensuring no convergence in 1st iteration

iter = 0;
while true
    iter=iter+1;

    newgamma=repmat(alpha,N,1);
    sumgamma=sum(gammahat,2);
    e1=exp(psi(gammahat) - repmat(psi(sumgamma),1,K));  % exp(E_q log gamma_i)
    
    ll = 0;
    %ll = ll + sum(log(gammahat),1)*(alpha-1)';
    %ll = ll - sum(log(sum(gammahat,2))).*(sum(alpha)-K);
    %ll = ll - (sum(log(gamma(alpha)))-log(gamma(sum(alpha))))*N;
    ll = ll - sum(log(sum(e1,2))) * (2*N-2); % common denum for delta

    numB = zeros(K,K);
    denB = zeros(K,K);

    e2 = e1';
    e3 = e2(:)';
    Blin = B(:);
    nBlin = 1-Blin;
    for i=1:N
        delta =reshape(e2(:,i)*e3,[K*K,N])...
            .* (Blin*y(i,:)+nBlin*(1-y(i,:)));  % size K^2*N
        sum_delta = reshape(sum(delta,1),[N,1]);     % size N*1
        ll = ll + sum(log(sum_delta)) - log(sum_delta(i));
        rev_sum_delta = 1./sum_delta;
        rev_sum_delta(i) = 0;
        newgamma = newgamma+ reshape(sum(reshape(delta,[K,K,N]),1),[K,N])'...
            .* repmat(rev_sum_delta,[1,K]);     % newgamma(j,:)
        delta2 = reshape(delta*rev_sum_delta,[K,K]);% size K*K
        newgamma(i,:) = newgamma(i,:)+sum(delta2,2)';
        denB = denB+delta2;
        rev_sum_delta = rev_sum_delta.*y(i,:)';
        numB = numB+reshape(delta*rev_sum_delta,[K,K]);
    end

    % update gammahat
    gammahat = newgamma;
    
    % update B
    B = (numB + 1e-10)./(denB + 2e-10);

    llchange = (ll-lastll)/lastll;
    lastll = ll;
    fprintf('iter=%d\t%f\t%f\n', iter, ll, llchange);
    if (abs(llchange) < 1e-6)
        break;
    end
    if (iter >= maxit)
        fprintf('exceed iteration limit %d\n', maxit);
        break;
    end
end
