function [] = analyzeClusters(pi,K, min_size)

pi_data = pi(:,2:end);
sum_pi = sum(pi_data,1);
plot(1:K, sum_pi);

indices = find(sum_pi<min_size)
sum_pi(indices)
pi_data_temp=pi_data(:,indices(2));
row_indx = find(pi_data_temp>0.50)

pi(row_indx,1)

end