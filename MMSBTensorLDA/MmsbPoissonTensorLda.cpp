/*
 * MmsbPoissonTensorLda.cpp
 *
 *  Created on: Feb 23, 2013
 *      Author: abhimank
 */

#include "armadillo"
#include <iostream>
#include <string>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>

using namespace arma;
using namespace std;
using namespace boost::math;
//TODO: Garbage collect?

//typedef unsigned short    u16;
typedef Cube<s16> s16cube;



class VariableClass{
public:
	fmat Y; 	// csv_ascii	//    self->Y=Y;
	int K;	//    self.K=K;
	int sampleIters;	//    self->sampleIters = sampleIters;
	int outer;	//    self->outer = outer;
	int L;	//    self->L = L;
	string filename;	//    self->filename = filename;
	fmat eta;	//    // prior for B	// uniform prior
	fmat alpha;	//    self->alpha = np.ones(K)*0.1; // uniform prior
	int num_users;	//    self->num_users = Y.shape[0];
	float threshold;	//    self.threshold = 1e-4;
	//    // Block Matrix count
	icube Nkk;	//    self->Nkk=np.zeros((K,K,2));       //tensor since it emulates Beta distribution, cluster><cluster
	fmat pi;	//    self->pi = np.zeros((self.num_users,K));
	fmat B;	//    self->B = np.zeros((K,K));
	s16cube Zuu;	//    self->Zuu = np.zeros((self.num_users,self.num_users,2)); // user><user stores index of cluster, z_pq; two sets to distinguish between z_p->q and z_q<-p
	imat Nuk;	//    self->Nuk = np.zeros((self.num_users, K));  // user><cluster, n_user_k
	fmat ll;	//    self->ll=np.zeros((sampleIters+outer,1));
};

void fillInitialVals(VariableClass* self);
Mat<s16> multinomial_bivariate_sample(VariableClass* self, fmat mult_probs);
float calculate_joint_log_likelihood(VariableClass* self);
void sampler(VariableClass* self);
fmat calculate_Mult_Probs(VariableClass* self, imat Npk, imat Nqk, int y_pq, int curr_p, int curr_q);

float randomGen(void){
	boost::mt19937 rng;
	static boost::uniform_01<boost::mt19937> zeroone(rng);
	return zeroone();;
}

void initialize(VariableClass* self, fmat Y, int K, int sampleIters, int outer, int L, string filename){

	self->Y=Y;
	self->K=K;
	self->sampleIters = sampleIters;
	self->outer = outer;
	self->L = L;
	self->filename = filename;

	// prior for B
	self->eta = ones<fmat>(1,2)*0.1;//np.ones(2)*0.1; // uniform prior
	// prior for pi
	self->alpha = ones<fmat>(1,K)*0.1;//np.ones(K)*0.1; // uniform prior

	self->num_users = Y.n_rows;//Y.shape[0];
//	cout<<"num_user "<<self->num_users<<endl;
	self->threshold = 1e-4;
	// Block Matrix count
	//TODO: I wonder whether a 3D zeros would work ==> works
	self->Nkk=zeros<icube>(K,K,2);//np.zeros((K,K,2));       //tensor since it emulates Beta distribution, cluster><cluster
	self->pi = zeros<fmat>(self->num_users,K);//np.zeros((self->num_users,K));

	self->B = zeros<fmat>(K,K);//np.zeros((K,K));
	//TODO: I wonder whether a 3D ones would work ==> works
	self->Zuu = ones<s16cube>(self->num_users,self->num_users,2)*-1;//np.zeros((self.num_users,self.num_users,2)); // user><user stores index of cluster, z_pq; two sets to distinguish between z_p->q and z_q<-p
	self->Nuk = zeros<imat>(self->num_users, K);//np.zeros((self.num_users, K));  // user><cluster, n_user_k
	self->ll= zeros<fmat>(sampleIters+outer,1);//np.zeros((sampleIters+outer,1));
//	cout<< self->Y<<endl;
//	cout<< self->Zuu<<endl;

	fillInitialVals(self);
//	cout<< self->Zuu<<endl;
}
void fillInitialVals(VariableClass* self){
	cout<< "Nkk sum "<<accu(self->Nkk)<<endl;//np.sum(self.Nkk);
	cout << "Nuk sum " <<accu(self->Nuk)<<endl; //np.sum(self.Nuk);
	for(int user_p=0; user_p<self->num_users;user_p++){
		for(int user_q=0; user_q<self->num_users;user_q++){
//			cout<<user_p<<" "<<user_q<<endl;
			if (user_p==user_q)
				continue;
			// sample Z_pq and Z_qp together
			//TODO: Deal with functions returning tuples
			fmat uniform_dist = ones<fmat>(self->K*self->K,1)*1.0/(self->K*self->K*1.0);
			Mat<s16> p_kq_k = multinomial_bivariate_sample(self, uniform_dist);//multinomial_bivariate_sample(np.ones((self->K*self->K,1))/(self->K*self->K));
			s16 p_k=p_kq_k(0);
			s16 q_k=p_kq_k(1);
//			cout<<"user_p, user_q, p_k, q_k "<<user_p<<" "<<user_q<<" "<<p_k<<" "<<q_k<<endl;
			self->Zuu(user_p,user_q,0) = p_k;
//			cout<<"first"<<self->Zuu<<endl;
			self->Zuu(user_q,user_p,1) = q_k;
//			cout<<"second"<<self->Zuu<<endl;
			int y_pq = self->Y(user_p,user_q); // point to note: y is 0,1, we don't need 1,2 as in Matlab
			self->Nkk(p_k,q_k,y_pq) = self->Nkk(p_k,q_k,y_pq)+1; // increment Ngh, y_pq
			self->Nuk(user_p,p_k) = self->Nuk(user_p,p_k)+1;
			self->Nuk(user_q,q_k) = self->Nuk(user_q,q_k)+1;

		}
	}

}

int getParameters(VariableClass* self){
	// TO BE NOTED send sampling probabilities only in the order they are expected.

	// Gibbs Sampling
	// At present going for fixed number of iterations

	self->pi = zeros<fmat>(self->num_users,self->K);//np.zeros((self.num_users,self.K));
	int inner_iter = 0;
	for(inner_iter=0;  inner_iter<self->sampleIters; inner_iter++){  //start from 0
		cout<<inner_iter<<endl;
		sampler(self);
		self->ll(inner_iter)= calculate_joint_log_likelihood(self);
//		cout<< "Nkk sum "<<accu(self->Nkk)<<endl;
//		cout << "Nuk sum " <<accu(self->Nuk)<<endl;
		cout<< self->ll(inner_iter)<<endl;
//		cout<< self->ll<<endl;
		//            if inner_iter>1 && abs(ll(inner_iter)-ll(inner_iter-1))<self.threshold :
		//                break;
	}

	fmat temp_pi=zeros<fmat>(self->num_users,self->K);//np.zeros((self.num_users,self.K));
	for(int iter=0; iter< self->outer; iter++){
		for(int inner=0; inner<self->L; inner++){
			//TODO: Deal with functions returning tuples
//			self->Nkk, self.Nuk, self.Zuu =
			sampler(self);
		}
		cout<< iter<<endl;

//		cout<<self->ll.n_elem<<" "<<self->ll.n_cols<<" "<<inner_iter<<endl;
//		cout<<self->ll<<endl;
		self->ll(inner_iter)=calculate_joint_log_likelihood(self);
		cout<< self->ll(inner_iter)<<endl;

		inner_iter = inner_iter+1;
		// estimate Pi
		for(int u=0; u<self->num_users; u++){
			for(int k=0; k<self->K; k++){
				temp_pi(u,k)=self->Nuk(u,k)+self->alpha(k);
			}
		}
		//TODO: write matrix repmat and reshape operations or use some other array trick
//		temp_pi=temp_pi/np.tile(np.sum(temp_pi,1).reshape(temp_pi.shape[0],1),(1,self.K));
		temp_pi = 1.0*temp_pi/repmat(sum(temp_pi,1),1,self->K);

		self->pi=self->pi+temp_pi;

		// estimate B, block matrix
		for(int g=0; g<self->K; g++)
			for(int h=0; h<self->K; h++)
				self->B(g,h)= self->B(g,h) + (self->Nkk(g,h,1)+self->eta(1))/(self->Nkk(g,h,1)+self->Nkk(g,h,0)+accu(self->eta));
	}
	cout<<"helloiamhere\n";
	self->pi=1.0*self->pi/self->outer;
	cout<< self->pi;
	self->B=1.0*self->B/self->outer;
	string temp_file = self->filename;
	self->pi.save(temp_file.append("_PI.csv"), csv_ascii);
	temp_file = self->filename;
	cout<< inner_iter;
	self->ll.save(temp_file.append("_LL.csv"), csv_ascii);
	return 1;
}

void sampler(VariableClass* self){
	for(int user_p=0;user_p<self->num_users; user_p++){
		for(int user_q=0;user_q<self->num_users; user_q++){
			if(user_p==user_q)       //if not then we will be sampling p->q twice. Wrong : y_pq != Y_qp
				continue;
//			cout<<"sampler0 "<<user_p<<" "<<user_q<<" "<<"num_users "<< self->num_users;
//			cout<<self->Zuu(user_p,user_q,0)<<endl;
			// get current assignment of k to p->q and q->p
			int p_k = (int) self->Zuu(user_p,user_q,0);
			int q_k = (int) self->Zuu(user_q,user_p,1);
//			cout<<"sampler1 "<<user_p<<" "<<user_q<<endl;
			int y_pq = self->Y(user_p,user_q);  // point to note: y is 0,1, we don't need 1,2
//			cout<< y_pq<< p_k<< q_k<< endl;

			fmat multi_probs = calculate_Mult_Probs(self, self->Nuk.row(user_p), self->Nuk.row(user_q), y_pq, p_k, q_k);

			// decrement current counts
			self->Nkk(p_k,q_k,y_pq) = self->Nkk(p_k,q_k,y_pq) - 1; // decrement Ngh,y_pq
			self->Nuk(user_p,p_k) = self->Nuk(user_p,p_k) - 1;
			self->Nuk(user_q,q_k) = self->Nuk(user_q,q_k) - 1;

			// sample Z_pq and Z_qp together
			Mat<s16> p_kq_k = multinomial_bivariate_sample(self, multi_probs);
			p_k = p_kq_k(0);
			q_k = p_kq_k(1);
			self->Zuu(user_p,user_q,0) = p_k;
			self->Zuu(user_q,user_p,1) = q_k;
			self->Nkk(p_k,q_k,y_pq) = self->Nkk(p_k,q_k,y_pq) + 1; // increment Ngh,y_pq
			self->Nuk(user_p,p_k) = self->Nuk(user_p,p_k) + 1;
			self->Nuk(user_q,q_k) = self->Nuk(user_q,q_k) + 1;
		}
	}
//	cout<<"Nuk "<<self->Nuk<<endl;
//	cout<<"Nkk "<<self->Nkk<<endl;
}

//    matrix or vector return?
fmat calculate_Mult_Probs(VariableClass* self, imat Npk, imat Nqk, int y_pq,
		int curr_p, int curr_q){
	fmat multi_probs= zeros<fmat>(self->K*self->K,1);
	int indx=0;
	float sum_eta=accu(self->eta);
//	cout<<"Npk "<<Npk<<endl;
	for(int  p_k=0; p_k<self->K; p_k++){
		for(int q_k=0; q_k<self->K; q_k++){
			float alpha_p=self->alpha(p_k);
			float alpha_q=self->alpha(q_k);
			float eta_local=self->eta(y_pq);
			if(p_k==curr_p and q_k==curr_q){
				eta_local = eta_local-1;
				sum_eta = sum_eta-1;
				alpha_p = alpha_p-1;
				alpha_q = alpha_q-1;
			}
//			cout<<"calculate_Mult_Probs "<<p_k<<" "<<q_k<<endl;
			multi_probs(indx)=((self->Nkk(p_k,q_k,y_pq)+eta_local)/((self->Nkk(p_k,q_k,1)+self->Nkk(p_k,q_k,0))+sum_eta));
			multi_probs(indx) = multi_probs(indx) * ((Npk(p_k)+alpha_p)) * ((Nqk(q_k)+alpha_q));
			indx=indx+1;
		}
	}
	return multi_probs;
}
//
//
//fmat vectorizeFloat(fmat input, float (*function)(float)){
//	fmat result(1,input.n_elem);
//	for(int i=0;i<input.n_elem;i++){
//		result(i)=(* function)(input(i));
//	}
//	return result;
//}


float calculate_joint_log_likelihood(VariableClass* self){
	float ll=0;
	//        cout<<self->Nuk;
	for(int g=0; g<self->K; g++){
		for(int h=0; h<self->K; h++){
			for(int y=0;y<2;y++)
				ll=ll+lgamma(self->Nkk(g,h,y)+self->eta(y))-lgamma(self->eta(y));
			//                    cout<<ll;
			ll=ll+boost::math::lgamma(1.0*accu(self->eta))-lgamma((self->Nkk(g,h,0)+self->Nkk(g,h,1))+accu(self->eta));
		}
	}
	for( int p=0; p<self->num_users; p++){
		for(int k=0; k<self->K; k++){
			ll=ll+lgamma(self->Nuk(p,k)+self->alpha(k))-lgamma(self->alpha(k));
		}
		float accuAlpha = accu(self->alpha);
		float accuNpk = accu(self->Nuk.row(p));
//		ll=ll+lgamma(accu(self->alpha))-lgamma(accu(self->Nuk.row(p))+accu(self->alpha));
		ll=ll+lgamma(accuAlpha)-lgamma(accuAlpha+accuNpk);
	}
	return ll;
}

Mat<s16> multinomial_bivariate_sample(VariableClass* self, fmat mult_probs){
	mult_probs = mult_probs/accu(mult_probs);
	mult_probs = cumsum(mult_probs,0);	// cum sum along the columns

//	cout<<mult_probs<<endl; //<< cumsum(mult_probs,1)<<endl;
	float rand_generated = randomGen();
	Mat<s16> k_pk_q(1,2);
	//        index = np.nonzero(mult_probs>np.random.random())[0][0];
	uvec indices = find(mult_probs>rand_generated);
	int index = indices(0);
	k_pk_q(0) = ceil(index/self->K);
	k_pk_q(1) = index%self->K;

	//        if k_q==0:    # this is not valid for python
	//        k_q=self.K
//	cout<<k_pk_q<<endl;
	return k_pk_q;
}

int main(){
    fmat a;//sio.loadmat("monkLK.mat", {})
    string filename = "abhimanu_sp_matrix_V2_5000_binary.csv";
    a.load(filename, csv_ascii);
    a=a.submat(0,0,999,999);
//    print a['y']
//    cls = MMSB(a['y'], 3, 1000, 100, 2, 'test_monk_python')
    VariableClass *self = new VariableClass();
    initialize(self, a, 3, 3000, 500, 5, filename);
//    cout<<"cls num_user "<<self->num_users<<endl;
//	cout<< "Nkk sum "<<accu(self->Nkk)<<endl;//np.sum(self.Nkk);
//	cout << "Nuk sum " <<accu(self->Nuk)<<endl; //np.sum(self.Nuk);
//	cout<<"done filling initial values"<<endl;
//	cout<<"num_user "<<self->num_users<<endl;
//	cout<<"Nuk "<< self->Nuk<<endl;
//	cout<<"Zuu "<< self->Zuu<<endl;
    getParameters(self);
}
