function [ralpha, rB, rll, rpi, gammahat] = MMSB_Learn_Orig(y, K, maxit)
% y: N*N
% ralpha: 1*K
% rB: K*K
% rpi: N*K
% gammahat: N*K

if (maxit <= 0); maxit=100; end
N=size(y,1);

ralpha = 0.5+(rand(1,K)-0.5)*0.1
rB = eye(K);
%rB = rand(K,K)
gammahat = repmat(ralpha,N,1)+(rand(N,K)-0.5).*0.1;


rll = [];
lastll = 1;
iter_l = 0;
while true
    iter_l=iter_l+1;

    [gammahat,rB,ll]=MMSB_Learn_InnerLoop_Orig(y,ralpha,rB,gammahat,5);

    % update alpha
    g = N.*(psi(sum(ralpha)) - psi(ralpha)) + sum(gammahat,1) - sum(psi(sum(gammahat,2))); % size 1*K
    H = N.*(diag(psi(1,ralpha)) - psi(1,sum(ralpha)));
    ralpha = ralpha+g*inv(H)*1e-5;
    
    % show progress
    ralpha;
    %rB

    rll = [rll ll];
    
    llchange = (ll-lastll)/lastll;
    lastll = ll;
%     fprintf('learn_iter=%d\t%f\t%f\n', iter_l, ll, llchange);
    if (abs(llchange) < 1e-6)
        break;
    end
    if (iter_l >= maxit)
%         fprintf('exceed iteration limit of learning %d\n', maxit);
        break;
    end
end

rpi = gammahat./repmat(sum(gammahat,2),1,K);
