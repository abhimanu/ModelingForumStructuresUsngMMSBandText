import numpy as np;
import scipy as sp;

numThreads = 10;
numUsers = 100;
K=10

# hyperparameters
alpha = np.ones(K)*0.1 #[0.1,0.1,0.1,0.1,0.1]
kappa = np.ones((K,K))
theta = np.ones((K,K))

fopen = open("syntheticData.txt", 'w')
fopenPi = open("syntheticData_pi.txt", 'w')

# parameters
pi = np.zeros((numUsers,K))
B = np.zeros((K,K))

for k1 in xrange(0,K):
	for k2 in xrange(0,K):
		kappa[k1,k2] = 0.5
		theta[k1,k2] = 0.5
		B[k1,k2] = np.random.gamma(0.5,0.5)
		if k1==k2:
			kappa[k1,k2] = 1
			theta[k1,k2] = 1
			B[k1,k2] = np.random.gamma(1,1)
			

for u in xrange(0, numUsers):
	pi[u,:] = np.random.dirichlet(alpha)
	fopenPi.write(str(u))
	for k in xrange(0,K):
		fopenPi.write(","+str(pi[u,k]))
	fopenPi.write("\n")


for i in xrange(0, numThreads):
	for p in xrange(0, numUsers):
		for q in xrange(0, numUsers):
			if(p==q):
				continue
			z_pq = np.random.multinomial(1,pi[p,:])
			z_qp = np.random.multinomial(1,pi[q,:])
			b = np.mat(z_pq)*np.mat(B)*(np.mat(z_qp).T)
			y_tpq = np.random.poisson(b)
			for it in xrange(0,y_tpq):
				fopen.write(str(p)+" "+str(q)+" "+str(i)+" 10000\n")
			# write in the file directly
