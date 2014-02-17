function [nA] = getNoisyVersion(A,noiseThresh)
nA=A;
[r,c] = size(A)
for i=1:r
    for j=1:c
        if rand > noiseThresh
            if nA(i,j)==0
                nA(i,j)=1;
            else
                nA(i,j)=0;
            end
        end
        
    end
end

end