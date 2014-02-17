function [] = syncKKtest()

K = 2;

num_users = 2;

Zuu1 = rand(num_users,num_users,2);
Zuu1(Zuu1>0.5) = 2;
Zuu1(Zuu1<=0.5) = 1;

Zuu2 = rand(num_users,num_users,2);
Zuu2(Zuu2>0.5) = 2;
Zuu2(Zuu2<=0.5) = 1;
 
Zuu1
Zuu2

Nkk_temp=zeros(K,K,2);
for y = 1:2
    Z1 = Zuu1(:,:,y);
    Z2 = Zuu2(:,:,y);
    for g = 1:K
        for h  = 1:K
            [r1,c1] = find(Z1==g);
            Z2_2 = Z2(sub2ind(size(Z2),c1,r1)); % p->q become q->p
            r = r1(Z2_2==h);
%           c = c1(Z2_2==h);
            Nkk_temp(g,h,y) = size(r,1);
            if g==1 && h==1
                Z1
                Z2
                r1
                c1
                Z2_2
                r
            end
        end
    end
end

Nkk_temp
% Zuu1(:,:,1)
% Zuu1(1,1,1)
sum(sum(sum(Nkk_temp)))
end
