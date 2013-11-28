load abhimanu_sp_matrix_V2.mat;
sp=arr(1:1000,1:1000);
load fm_counts_test_V2.mat
fm=arr(1:1000,1:1000);
[alpha, B1,B2, pi, eta] = MMSB_sampling_poisson_tensorNew(sp(901:1000,901:1000),fm(901:1000,901:1000),3,1500,500,1,'sp_fm_tensor_poisson_901-1000_1500_500_3-3');
