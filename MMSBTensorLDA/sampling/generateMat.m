function [] = generateMat(length)
x=zeros(length,length);

numC = 3;

offset = length/3;
for c = 1:numC
    for i = 1:offset
        for j = 1:offset
            if i==j
                continue;
            end
            x((c-1)*offset+i,(c-1)*offset+j) = rand*20;
        end
    end
end

save(strcat('18','_simulatedMat.mat'),'x');

end