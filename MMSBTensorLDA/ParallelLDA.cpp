
#include "armadillo"
#include <iostream>
#include <string>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>
#include <mpi.h>
#include <vector>

#define MASTER_RANK 0
#define TAG_BASE 1000
#define C_LOG_LIMIT -740

using namespace arma;
using namespace std;
using namespace boost::math;
//TODO: Garbage collect?

//typedef unsigned short    u16;
typedef Cube<s16> s16cube;
typedef Mat<s16> s16mat;


class VariableClass{
public:
	//slave and master have different data dimensions
//	umat Y1; 	// csv_ascii
//	umat Y2; 	// csv_ascii
	umat W; 	// csv_ascii
	int K;	//    self.K=K;

	// these should be in all classes
	unsigned int num_users;	//    self->num_users = Y.shape[0];
	int vocab_size; //= self.W.shape[1] # vocab_size;

	//TODO: check for assigned_users

	//these should be in all classes
	//MPI related structure
	int num_slaves; //= self.num_procs-1
	int slave_width; //= self.Y1.shape[0]/num_slaves
	int rank;
	int num_procs;

	//these should be in all classes except threshold
	// Constants/Priors
	float eta;	//    // prior for B	// uniform prior
	fmat alpha;	//    self->alpha = np.ones(K)*0.1; // uniform prior
	float kappa;
	float beta;

};

class MasterClass: public VariableClass{
public:
	// these should be only in master class
	int sampleIters;	//    self->sampleIters = sampleIters;
	int outer;	//    self->outer = outer;
	int L;	//    self->L = L;
	string filenameSave;	//    self->filename = filename;

	//these should be in class specific as the dimensions are different
	// Block Matrix count: Root maintains Sum_kk1 for each
//	ucube Sum_kk1;	//=np.zeros((num_slaves,self.K,self.K)); # needed by poisson distribution sampling
//	ucube Sum_kk2;	//=np.zeros((num_slaves,self.K,self.K)); # needed by poisson distribution sampling
	s16mat Zuu1; 		//= np.ones((self.num_users,self.num_users), dtype=np.int16)*-1; # user><user stores index of cluster, z_pq; two sets to distinguish between z_p->q and z_q<-p
	s16mat Zuu2; 		//= np.ones((self.num_users,self.num_users), dtype=np.int16)*-1; # user><user stores index of cluster, z_pq; two sets to distinguish between z_p->q and z_q<-p
	ucube Nuk; 		//= np.zeros((num_slaves, self.num_users, self.K));  # user><cluster, n_user_k
//	ucube Nkk;		//=np.zeros((num_slaves,self.K,self.K));   # it emulates gamma distribution? cluster><cluster
	// LDA initializations
	ucube Nkw; 		//= np.zeros((num_slaves,self.K,self.vocab_size), dtype=np.uint32)

	// these should only be in master
	// Parameters learned
	fmat pi;	//    self->pi = np.zeros((self.num_users,K));
	fmat phi;	//    self->pi = np.zeros((self.num_users,K));
//	fmat B;	//    self->B = np.zeros((K,K));
	fmat ll;	//    self->ll=np.zeros((sampleIters+outer,1));
	float threshold ;	//    self.threshold = 1e-4;

};


class SlaveClass: public VariableClass{
public:
	//these should be in class specific as the dimensions are different
	// Block Matrix count: Root maintains Sum_kk1 for each
	umat Sum_kk1;	//=np.zeros((num_slaves,self.K,self.K)); # needed by poisson distribution sampling
	umat Sum_kk2;	//=np.zeros((num_slaves,self.K,self.K)); # needed by poisson distribution sampling

	umat Sum_kk1_p;	//=np.zeros((num_slaves,self.K,self.K)); # needed by poisson distribution sampling
	umat Sum_kk2_p;

	s16mat Zuu1; 		//= np.ones((self.num_users,self.num_users), dtype=np.int16)*-1; # user><user stores index of cluster, z_pq; two sets to distinguish between z_p->q and z_q<-p
	s16mat Zuu2; 		//= np.ones((self.num_users,self.num_users), dtype=np.int16)*-1; # user><user stores index of cluster, z_pq; two sets to distinguish between z_p->q and z_q<-p
//	s16mat Zuu1_p; 		//= np.ones((self.num_users,self.num_users), dtype=np.int16)*-1; # user><user stores index of cluster, z_pq; two sets to distinguish between z_p->q and z_q<-p
//	s16mat Zuu2_p;

	umat Nuk; 		//= np.zeros((num_slaves, self.num_users, self.K));  # user><cluster, n_user_k
	umat Nkk;		//=np.zeros((num_slaves,self.K,self.K));   # it emulates gamma distribution? cluster><cluster
	umat Nuk_p; 		//= np.zeros((num_slaves, self.num_users, self.K));  # user><cluster, n_user_k
	umat Nkk_p;
	// LDA initializations
	umat Nkw; 		//= np.zeros((num_slaves,self.K,self.vocab_size), dtype=np.uint32)
	umat Nkw_p; 		//= np.zeros((num_slaves,self.K,self.vocab_size), dtype=np.uint32)
	vector< s16mat > Zuw;

	int numLocalUpdateVars;
};


void runComputation(VariableClass* self);
void fillInitialVals(VariableClass* self);
Mat<s16> multinomial_bivariate_sample(VariableClass* self, fmat mult_probs);
void slaveSampler(SlaveClass* self);
//fmat calculate_Mult_Probs(VariableClass* self, imat Npk, imat Nqk, int y_pq, int curr_p, int curr_q);
void masterIntialization(MasterClass* self, string* filenamesRead, string filenameSave, int sampleIters, int outer, int L, int assigned_users);
void slaveInitialization(SlaveClass *self);
int lda_multinomial_sample(SlaveClass* self,fmat mult_probs);
void receiveAndSyncFromSlaves(MasterClass* self);
void sendLocalUpdatesToMaster(SlaveClass* self);
void getFirstGlobalUpdatesFromMaster(SlaveClass* self);
umat getTheSum(ucube cube_full);
void sendGlobalupdatesToSlaves(MasterClass *self, int end_tag);
MPI_Request getGlobalUpdatesFromMaster(SlaveClass* self);
int getMasterParameters(MasterClass* self);
float log_nbinpdf(int k, int r, float pr);
fmat calculate_Mult_Probs_mmsbZ(SlaveClass* self,umat Npk, umat Nqk,int y_pq1, int y_pq2, int curr_p, int curr_q);
fmat calculate_Mult_Probs_ldaZ(SlaveClass* self, umat Npk, int curr_p, int word);
fmat vectorizeFloat(fmat input, float (*function)(float));
fmat gammaln(fmat input);
float getSumLnGammaY_pq(MasterClass* self, int g, int h, umat Y);
void slaveComputation(SlaveClass* self);
void fillinInitialSlaveVals(SlaveClass* self);
void saveFiles(MasterClass* self, int iter);
umat getTheSum(ucube cube_full);
float calculate_joint_log_likelihood(MasterClass* self);



float randomGen(void){
	boost::mt19937 rng;
	static boost::uniform_01<boost::mt19937> zeroone(rng);
	return zeroone();;
}

//self, filenamesRead, filenameSave, sampleIters, outer, L, dictName, assigned_users
void initialize(VariableClass* self, string* filenamesRead, int K, int sampleIters, int outer, int L, int assigned_users, string filenameSave){
	self->K=K;

	// prior for B
	self->eta = 10.0; // uniform prior
	self->kappa = 10.0;
	// prior for pi
	self->alpha = ones<fmat>(1,K)*0.1;//np.ones(K)*0.1; // uniform prior
	// prior for LDA
	self->beta = 1.0;

	// get MPI setup informations, number of processors, current processor etc.
	MPI_Comm_rank(MPI_COMM_WORLD, &(self->rank));
	cout<< "my rank "<<self->rank<<endl;
	MPI_Comm_size(MPI_COMM_WORLD, &(self->num_procs));
	self->num_slaves = self->num_procs-1;


//	const int masterRank = MASTER_RANK;
	//TODO: is use of macro as integer ok?
	if (self->rank == MASTER_RANK){
		// initialize master; we donot send anything to the slaves in Asynch version
		masterIntialization((MasterClass *)self, filenamesRead,filenameSave, sampleIters, outer, L, assigned_users);
		// wait for init acks, these acks are dangerous here and shud be done inmasterIntialization
		//            for s in xrange(1, self.num_procs):
		//                ack = self.com.recv()
	}

	else{ // SLAVES
		// initialize slaves
		slaveInitialization((SlaveClass *) self);
		// no need to send acks as the aggregated values are being sent by fillInitialSlaves
		//            self.com.send( 1,dest=0, tag=0) # acknowledge that init is done
		//            self.com.isend( dest=0, tag=self.rank, 0) # acknowledge that init is done
	}

//	fillInitialVals(self);
//	cout<< self->Zuu<<endl;
}


void runComputation(VariableClass* self){
	//TODO: is use of macro as integer ok?
    if (self->rank==MASTER_RANK)  // master
        getMasterParameters((MasterClass*) self); // master does all the computations here
        // stop, print and save shit
    else{                       // slaves
        cout<<"SLAVE: slave in runComputation, rank "<<self->rank;
        slaveComputation((SlaveClass*) self);
    }
}



void masterIntialization(MasterClass* self, string* filenamesRead, string filenameSave, int sampleIters, int outer, int L, int assigned_users){

	cout<<"At the top of Master Initialization"<<endl;
    self->sampleIters = sampleIters;
    cout<<"MASTER: sampleIters "<<sampleIters<<endl;
    self->outer = outer;
    self->L = L;
    self->filenameSave = filenameSave;
    cout<<"MASTER: filenameSave "<<endl;


    self->ll=zeros<fmat>(sampleIters+outer,1);
    self->threshold = 1e-4;
//    string dir = "usr0/home/abhimank/CMU/fall2012/ML701/project/MMSBsimple/mpi/mmsb_lda/";
    //TODO: check for assigned_users
    //TODO:DONE what if there are more than 3 files to read.
    cout<<"MASTER: filenamesRead "<<filenamesRead<<endl;
//    self->Y1.load(filenamesRead[0], csv_ascii);
//    self->Y1 = self->Y1.submat(0,0,assigned_users-1,assigned_users-1);
//    cout<<"MASTER: Loaded Y1 from file"<<endl;
//    cout<< "MASTER: rank, self->Y1 "<<self->rank<<" "<< self->Y1<<endl;

//	self->Y2.load(filenamesRead[1], csv_ascii);
//	self->Y2 = self->Y2.submat(0,0,assigned_users-1,assigned_users-1);
//	cout<< "MASTER: rank, self->Y2 "<<self->rank<<" "<< self->Y2<<endl;

	self->W.load(filenamesRead[2], csv_ascii);
	cout<<"MASTER: self->W(0,208) "<<self->W(0,208)<<endl;
	self->W = self->W.rows(0, assigned_users-1);
	cout<<"MASTER: self->W(0,208) "<<self->W(0,208)<<endl;
//    self.Y1=np.array(np.squeeze(np.load(filenamesRead[0])), np.uint16)[0:assigned_users,0:assigned_users]#np.array(Y1[0:10,0:10],np.uint16);#Y[0:10,0:10];
//    self.Y2=np.array(np.squeeze(np.load(filenamesRead[1])), np.uint16)[0:assigned_users,0:assigned_users]#np.array(Y2[0:10,0:10],np.uint16);
//    self.W=np.array(np.squeeze(np.load(filenamesRead[2])), np.uint16)[0:assigned_users,:]#np.array(W[0:10,:], np.uint16)#.astype(int);   # user >< vocab-size

    cout<< "MASTER: Total num of  words: "<< accu(self->W)<<endl;

    self->num_users = self->W.n_rows;
    self->slave_width = self->num_users/self->num_slaves;

    self->vocab_size = self->W.n_cols;//shape[1] # vocab_size;
    cout<<"num_users, slave_width, W.shape "<<  self->num_users<<self->slave_width << self->W.n_rows << self->W.n_cols<<endl;
    // print self.W[0:self.num_users/2,:]
    // Block Matrix count: Root maintains Sum_kk1 for each
    //NOTE: the way 3D arrays are in armadillo and python/matlab, use of slices; be careful in synching and summing.
//    self->Sum_kk1=zeros<ucube>(self->K,self->K,self->num_slaves); // needed by poisson distribution sampling
//    self->Sum_kk2=zeros<ucube>(self->K,self->K,self->num_slaves); // needed by poisson distribution sampling

    // self->Zuu not needed
//    self->Zuu1 = ones<s16mat>(self->num_users,self->num_users)*-1; // user><user stores index of cluster, z_pq; two sets to distinguish between z_p->q and z_q<-p
//    self->Zuu2 = ones<s16mat>(self->num_users,self->num_users)*-1; // user><user stores index of cluster, z_pq; two sets to distinguish between z_p->q and z_q<-p

    //NOTE: the way 3D arrays are in armadillo and python/matlab, use of slices;
    self->Nuk = zeros<ucube>(self->num_users, self->K, self->num_slaves);  // user><cluster, n_user_k
//    self->Nkk=zeros<ucube>(self->K,self->K,self->num_slaves);   // it emulates gamma distribution? cluster><cluster

    // LDA initializations
    //NOTE: the way 3D arrays are in armadillo and python/matlab, use of slices;
    self->Nkw = zeros<ucube>(self->K,self->vocab_size,self->num_slaves);

    unsigned int init_msg[] = {self->num_users, self->slave_width, self->vocab_size, self->num_slaves};
    // send the data to slaves
    // FIRST SEND by any process; non-blocking send
    // Slaves have to fetch this message via tags, careful.
    MPI_Request r; int tag=0;
    for(int s=1; s<self->num_procs; s++){ // num_slave = num_procs-1
//            print i, self.W[(i-1)*self.slave_width:i*self.slave_width,:]
    	int start_index = (s-1)*self->slave_width;
    	int end_index = (s)*self->slave_width;
    	//send num_user and slave_width
    	tag=1;
    	MPI_Isend(init_msg,sizeof(init_msg)/sizeof(init_msg[0]),MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&r);
    	cout<<"MASTER: After SENDING init_msg MASTER Initialization"<<endl;

    	//send Y1
    	tag++;
    	umat sub_y1 = self->Y1.rows(start_index,end_index-1);
    	MPI_Isend(&(sub_y1(0)),sub_y1.n_elem,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&r);
    	cout<<"MASTER: After SENDING Y1 MASTER Initialization"<<endl;

    	//send Y2
    	tag++;
    	umat sub_y2 = self->Y2.rows(start_index,end_index-1);
    	MPI_Isend(&(sub_y2(0)),sub_y2.n_elem,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&r);
    	cout<<"MASTER: After SENDING W MASTER Initialization"<<endl;

    	//send W
    	tag++;
    	umat sub_w = self->W.rows(start_index,end_index-1);
    	cout<<"MASTER: sub_w(0,208) "<<sub_w(0,208)<<endl;
    	MPI_Isend(&(sub_w(0)),sub_w.n_elem,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&r);
    	cout<<"MASTER: After SENDING W MASTER Initialization"<<endl;
//        self.com.isend(dest=i, tag=0, value=(self.slave_width,
//                       self.Y1[(i-1)*self.slave_width:i*self.slave_width,:],
//                       self.Y2[(i-1)*self.slave_width:i*self.slave_width,:],
//                       np.array(self.W[(i-1)*self.slave_width:i*self.slave_width,:], dtype=np.uint32)))
    }
    // receive data from slaves
    receiveAndSyncFromSlaves(self);
    sendGlobalupdatesToSlaves(self,0);
}


void slaveInitialization(SlaveClass *self){ // only slaves call slaveInitialization
    //rcv intials, blocking recv
	cout<<"SLAVE: At the top of Slave Initialization"<<endl;
	unsigned int init_msg[] ={0,0,0,0};
//	MPI_Request r;
	MPI_Status status;
	int tag=1;
	MPI_Recv(init_msg,sizeof(init_msg)/sizeof(init_msg[0]),MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&status);
	self->num_users=init_msg[0];
	self->slave_width=init_msg[1];
	self->vocab_size=init_msg[2];
	self->num_slaves=init_msg[3];
	self->num_procs=self->num_slaves+1;
	cout<<"SLAVE: After receiving init_msg Slave Initialization, rank "<< self->rank<<endl;
	cout<< "SLAVE: rank, init_msg "<<self->rank<<" "<< init_msg<<endl;
	//send Y1
	tag++;
	self->Y1 = zeros<umat>(self->slave_width,self->num_users);
	MPI_Recv(&(self->Y1(0)),self->Y1.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&status);
	cout<<"SLAVE: After receiving Y1 Slave Initialization, rank "<< self->rank<<endl;
	cout<< "SLAVE: rank, self->Y1 "<<self->rank<<" "<< self->Y1<<endl;

	//send Y2
	tag++;
	self->Y2 = zeros<umat>(self->slave_width,self->num_users);
	MPI_Recv(&(self->Y2(0)),self->Y2.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&status);
	cout<<"SLAVE: After receiving Y2 Slave Initialization, rank "<< self->rank<<endl;
	cout<< "SLAVE: rank, self->Y2 "<<self->rank<<" "<< self->Y2<<endl;

	//send W
	tag++;
	self->W = zeros<umat>(self->slave_width,self->vocab_size);
	MPI_Recv(&(self->W(0)),self->W.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&status);
	cout<<"SLAVE: After receiving W Slave Initialization, rank "<< self->rank<<endl;

	cout<<"SLAVE: received self->W(0,208), rank: "<<self->W(0,208)<<" "<<self->rank<<endl;

//    self.W = np.array(self.W, dtype=np.int16)   # save data space
    cout<< "SLAVE: Initial received Y1, Y2, ranks"<< accu(self->Y1)<<" "<< accu(self->Y2)<<" "<< self->rank<<endl;
//    self.num_users = self.Y1.shape[1] #self.Y.shape[0]        # this is wrong, num_users shoudl be Y[1]
//    self.vocab_size = self.W.shape[1] #vocab_size;
    fillinInitialSlaveVals(self);

    sendLocalUpdatesToMaster(self);
//    self.com.isend(dest=0, tag=1, value=(self.rank,self.Nuk,self.Nkk, self.Nkw, self.Sum_kk1, self.Sum_kk2, self.Zuu1, self.Zuu2))
    // this is different from later updates from master since this one is blocking
    getFirstGlobalUpdatesFromMaster(self);
    cout<<"SLAVE: get first global updates from master"<<endl;
}


void getFirstGlobalUpdatesFromMaster(SlaveClass* self){
	// self.Nuk,self.Nkk, self.Nkw, self.Sum_kk1, self.Sum_kk2
	// blocking
	MPI_Status status;

	// end_tag
	int tag=0, end_tag=0;
	MPI_Recv(&(end_tag),1,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&status);


	// self->Nuk_p
	tag=1;
	MPI_Recv(&(self->Nuk_p(0)),self->Nuk_p.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&status);

	//get self->Nkk_p
	tag++;
	MPI_Recv(&(self->Nkk_p(0)),self->Nkk_p.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&status);

	//get self->Nkw_p
	tag++;
	MPI_Recv(&(self->Nkw_p(0)),self->Nkw_p.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&status);

	//get self->Sum_kk1_p
	tag++;
	MPI_Recv(&(self->Sum_kk1_p(0)),self->Sum_kk1_p.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&status);

	//get self->Sum_kk1_p
	tag++;
	MPI_Recv(&(self->Sum_kk2_p(0)),self->Sum_kk2_p.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&status);

	cout<<"SLAVE: Received getFirstGlobalUpdatesFromMaster from master, rank "<<self->rank<<endl;
}


void sendGlobalupdatesToSlaves(MasterClass *self, int end_tag){
	MPI_Request r; int tag=1;
	for(int s=1; s<self->num_procs; s++){ // num_slave = num_procs-1
		//            print i, self.W[(i-1)*self.slave_width:i*self.slave_width,:]
		int start_index = (s-1)*self->slave_width;
		int end_index = (s)*self->slave_width;
		umat temp_mat;

		//send end_tag
		tag=0;
		MPI_Isend(&(end_tag),tag,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&r);


		//send Nuk
		tag=1;
		temp_mat = getTheSum(self->Nuk) - self->Nuk.slice(s-1);
		MPI_Isend(&(temp_mat(0)),temp_mat.n_elem,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&r);

		//send Nkk
		tag++;
		temp_mat = getTheSum(self->Nkk) - self->Nkk.slice(s-1);
		MPI_Isend(&(temp_mat(0)),temp_mat.n_elem,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&r);

		//send Nkw
		tag++;
		temp_mat = getTheSum(self->Nkw) - self->Nkw.slice(s-1);
		MPI_Isend(&(temp_mat(0)),temp_mat.n_elem,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&r);

		//send Sum_kk1
		tag++;
		temp_mat = getTheSum(self->Sum_kk1) - self->Sum_kk1.slice(s-1);
		MPI_Isend(&(temp_mat(0)),temp_mat.n_elem,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&r);

		//send Sum_kk2
		tag++;
		temp_mat = getTheSum(self->Sum_kk2) - self->Sum_kk2.slice(s-1);
		MPI_Isend(&(temp_mat(0)),temp_mat.n_elem,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&r);
	}
}


umat getTheSum(ucube cube_full){
	umat temp_mat = zeros<umat>(cube_full.n_rows, cube_full.n_cols);
	for(int i=0; i<cube_full.n_slices; i++)
		temp_mat+=cube_full.slice(i);
	return temp_mat;
}

void fillinInitialSlaveVals(SlaveClass* self){
	cout<< "SLAVE: Inside slave initials, ranks "<<self->rank<<endl;
	self->numLocalUpdateVars=5;	//self.Nuk,self.Nkk, self.Nkw, self.Sum_kk1, self.Sum_kk2
	// intialize Zs and Zufficient stats
	// Block Matrix count
	self->Sum_kk1=zeros<umat>(self->K,self->K); // needed by poisson distribution sampling
	self->Sum_kk2=zeros<umat>(self->K,self->K); // needed by poisson distribution sampling
	self->Sum_kk1_p=zeros<umat>(self->K,self->K); // needed by poisson distribution sampling
	self->Sum_kk2_p=zeros<umat>(self->K,self->K); // needed by poisson distribution sampling

	self->Zuu1 = ones<s16mat>(self->slave_width,self->num_users)*-1; // user><user stores index of cluster, z_pq; two sets to distinguish between z_p->q and z_q<-p
	self->Zuu2 = ones<s16mat>(self->num_users, self->slave_width)*-1; // user><user stores index of cluster, z_pq; two sets to distinguish between z_p->q and z_q<-p

	self->Nuk = zeros<umat>(self->num_users, self->K);  // user><cluster, n_user_k
	self->Nuk_p = zeros<umat>(self->num_users, self->K);  // user><cluster, n_user_k

	self->Nkk=zeros<umat>(self->K,self->K);   // it emulates gamma distribution? cluster><cluster
	self->Nkk_p=zeros<umat>(self->K,self->K);   // it emulates gamma distribution? cluster><cluster

	int start_index = (self->rank-1)*self->slave_width;
	int end_index = (self->rank)*self->slave_width;

	// LDA initializations
	self->Nkw = zeros<umat>(self->K,self->vocab_size);
	self->Nkw_p = zeros<umat>(self->K,self->vocab_size);
	self->Zuw.resize(self->num_users);
	cout<<"SLAVE: num_users, vocab_size, slave_width, rank "<<self->num_users<<" "<<self->vocab_size<<" "<<self->slave_width<< self->rank<<endl;

	//        for user_p in xrange(0,self.slave_width):
	//            user_p_orig = user_p + start_index
	//            dimension = np.sum(self.W[user_p,:])
	//            self.Zuw[user_p] = np.ones((dimension), dtype = np.int16)*-1

	//        total_count=0
	cout<<"SLAVE: fillinInitialSlaveVals initialized the vars going into for loop, rank "<<self->rank<<endl;
	int user_p_orig=0;
	for( int user_p=0; user_p<self->slave_width;user_p++){
		user_p_orig = user_p + start_index;
		for(int user_q=0; user_q<self->num_users; user_q++){
			if (user_p_orig==user_q)       //if not then we will be sampling p->q twice. Wrong : y_pq != Y_qp
				continue;
			// sample Z_pq and Z_qp together
			fmat uniform_dist = ones<fmat>(self->K*self->K,1)*1.0/(self->K*self->K*1.0);
			Mat<s16> p_kq_k = multinomial_bivariate_sample(self, uniform_dist);//multinomial_bivariate_sample(np.ones((self->K*self->K,1))/(self->K*self->K));
			s16 p_k=p_kq_k(0);
			s16 q_k=p_kq_k(1);
			self->Zuu1(user_p,user_q) = p_k; // 0,1 instead of matlab's 1,2
			self->Zuu2(user_q,user_p) = q_k;
			int y_pq1 = self->Y1(user_p,user_q); // point to note: y is 0,1, we don't need 1,2 as in Matlab
			int y_pq2 = self->Y2(user_p,user_q);
			self->Nkk(p_k,q_k) = self->Nkk(p_k,q_k)+1; // increment Ngh, y_pq

			self->Sum_kk1(p_k,q_k)=self->Sum_kk1(p_k,q_k)+y_pq1; // needed for poisson
			self->Sum_kk2(p_k,q_k)=self->Sum_kk2(p_k,q_k)+y_pq2;

			self->Nuk(user_p_orig,p_k) = self->Nuk(user_p_orig,p_k)+1;
			self->Nuk(user_q,q_k) = self->Nuk(user_q,q_k)+1;
		}
		cout<<"SLAVE: fillinInitialSlaveVals midway in the loop; LDA going to; rank "<<self->rank<<endl;
		// sample for LDA
		int dimension = accu(self->W.row(user_p));	//np.sum(self.W[user_p,:])
		self->Zuw[user_p] = ones<s16mat>(1,dimension)*-1;
		int cumul_count = 0;
		fmat uniform_dist_lda;
		try
		{
		for(int word=0; word< self->vocab_size; word++){
			unsigned int word_count = self->W(user_p,word);
			if(self->rank==1)
				cout<<"SLAVE: word, word_count, user_p, rank"<<" "<<word<<" "<<word_count<<" "<<user_p<<" "<<self->rank<<endl;
			if (word_count==0)
				continue;
			// print word_count
			for( int word_num = cumul_count;word_num< cumul_count+word_count; word_num++){
				uniform_dist_lda = ones<fmat>(self->K,1)*1.0/(self->K*1.0);
				int z_w = lda_multinomial_sample(self, uniform_dist_lda);
				self->Zuw[user_p](word_num) = z_w;     // word of Z and Nkw must not be the same
				self->Nkw(z_w,word)+=1;
				self->Nuk(user_p_orig,z_w)+=1;     // Zs from MMSB and LDA are piling up
			}
			cumul_count = cumul_count + word_count;
		//	if(self->)
		}
		}
		catch(int e){
			cout<<uniform_dist_lda<<" "<<self->rank<<"rank; in exception"<<endl;
		}
		//            total_count += cumul_count
		//        print "total_count", total_count
	}
	cout<<"SLAVE: finished fillinInitialSlaveVals, rank "<<self->rank<<endl;
}

void sendLocalUpdatesToMaster(SlaveClass* self){
	//Zuu are not really needed; needed for log likelihood computation
	// self.rank,self.Nuk,self.Nkk, self.Nkw, self.Sum_kk1, self.Sum_kk2, self.Zuu1, self.Zuu2))
	// Asynchronous sends
	MPI_Request r;
//	for(int i=1; i<=self->numLocalUpdateVars;i++){

	//self->Nuk
	int tag= 1; //self->rank*TAG_BASE+1; No need since master will have source info
	MPI_Isend(&(self->Nuk(0)),self->Nuk.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&r);

	//self->Nkk
	tag++;
	MPI_Isend(&(self->Nkk(0)),self->Nkk.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&r);

	//self->Nkw
	tag++;
	MPI_Isend(&(self->Nkw(0)),self->Nkw.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&r);

	//self->Sum_kk1
	tag++;
	MPI_Isend(&(self->Sum_kk1(0)),self->Sum_kk1.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&r);

	//self->Sum_kk2
	tag++;
	MPI_Isend(&(self->Sum_kk2(0)),self->Sum_kk2.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&r);

	//self->Zuu1
	tag++;
	MPI_Isend(&(self->Zuu1(0)),self->Zuu1.n_elem,MPI_SHORT,0,tag,MPI_COMM_WORLD,&r);

	//self->Zuu2
	tag++;
	MPI_Isend(&(self->Zuu2(0)),self->Zuu2.n_elem,MPI_SHORT,0,tag,MPI_COMM_WORLD,&r);


//	}
}


void receiveAndSyncFromSlaves(MasterClass* self){     // only master calls this: Remember the storing of processes' stats starts at 0
    cout<< "MASTER: In receiveAndSyncFromSlaves, rank "<< self->rank<<endl;
    s16mat temp_Z1 = zeros<s16mat>(self->slave_width, self->num_users);
    s16mat temp_Z2 = zeros<s16mat>(self->num_users, self->slave_width);
    int start_index, end_index;
    cout<<"MASTER: start_index, end_index "<<start_index<<" "<<end_index<<endl;
    for( int s=1; s<self->num_procs; s++){ // block receive
    	//sender_rank, Nuk_s, Nkk_s, Nkw_s, Sum_Nkk1_s, Sum_Nkk2_s, Zuu_s1, Zuu_s2= self.com.recv()
    	// update MMSB suff stats
    	MPI_Status status;
    	int tag=1;
        start_index = (s-1)*self->slave_width;
        end_index = (s)*self->slave_width;
    	MPI_Recv(&(self->Nuk.slice(s-1)(0)),self->Nuk.slice(s-1).n_elem,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&status);
        //self.Nuk[sender_rank-1,:,:] = Nuk_s

    	tag++;
    	MPI_Recv(&(self->Nkk.slice(s-1)(0)),self->Nkk.slice(s-1).n_elem,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&status);
        //self.Nkk[sender_rank-1,:,:] = Nkk_s

		// update LDA suff stats
    	tag++;
    	MPI_Recv(&(self->Nkw.slice(s-1)(0)),self->Nkw.slice(s-1).n_elem,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&status);
        //self.Nkw[sender_rank-1,:,:] = Nkw_s

    	tag++;
    	MPI_Recv(&(self->Sum_kk1.slice(s-1)(0)),self->Sum_kk1.slice(s-1).n_elem,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&status);
//    	self.Sum_kk1[sender_rank-1,:,:] = Sum_Nkk1_s

    	tag++;
    	MPI_Recv(&(self->Sum_kk2.slice(s-1)(0)),self->Sum_kk2.slice(s-1).n_elem,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&status);

    	tag++;
    	MPI_Recv(&(temp_Z1(0)),temp_Z1.n_elem,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&status);
        self->Zuu1.rows(start_index,end_index-1) = temp_Z1;

        tag++;
        MPI_Recv(&(temp_Z2(0)),temp_Z2.n_elem,MPI_UNSIGNED,s,tag,MPI_COMM_WORLD,&status);
        self->Zuu2.cols(start_index,end_index-1) = temp_Z2;

//        start_index = (sender_rank-1)*self.slave_width
//        end_index = (sender_rank)*self.slave_width
//        # Sync Zuu
//        self.Zuu1[start_index:end_index,:] = Zuu_s1 # just update the relevant part.
//        self.Zuu2[:,start_index:end_index] = Zuu_s2
    }
    cout<<"MASTER: At end of receiveAndSyncFromSlaves, rank"<< self->rank<<endl;
}


MPI_Request getGlobalUpdatesFromMaster(SlaveClass* self){
	// self.Nuk,self.Nkk, self.Nkw, self.Sum_kk1, self.Sum_kk2
	// blocking
	MPI_Request req[5];

	// move end_tag Recv in the slaveComputation
	// end_tag
//	int tag=0, end_tag=0;
//	MPI_Irecv(&(end_tag),1,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&(req[0]));

	// self->Nuk_p
	int tag=1;
	MPI_Irecv(&(self->Nuk_p(0)),self->Nuk_p.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&(req[0]));

	//get self->Nkk_p
	tag++;
	MPI_Irecv(&(self->Nkk_p(0)),self->Nkk_p.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&(req[1]));

	//get self->Nkw_p
	tag++;
	MPI_Irecv(&(self->Nkw_p(0)),self->Nkw_p.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&(req[2]));

	//get self->Sum_kk1_p
	tag++;
	MPI_Irecv(&(self->Sum_kk1_p(0)),self->Sum_kk1_p.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&(req[3]));

	//get self->Sum_kk1_p
	tag++;
	MPI_Irecv(&(self->Sum_kk2_p(0)),self->Sum_kk2_p.n_elem,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&(req[4]));

	cout<<"SLAVE: Received getFirstGlobalUpdatesFromMaster from master, rank "<<self->rank<<endl;

	return req[4];
}

void slaveComputation(SlaveClass* self){        // only slaves call this
	// Gibbs Sampling
	// At present going for fixed number of iterations
	// first1 = True // No need, We have already synched global values from Master
	int completedFlag=-1; // to signal whether earlier call to receive was answered so that req_obj is not overridden
	MPI_Request req_object, req_end_tag;
	int end_tag=0;
	MPI_Status stat;
	while(end_tag>=0){
		// compute new Zuu and Nuk first called after initialization
		slaveSampler(self);
		if(completedFlag<0 || completedFlag){ // send and receive
			sendLocalUpdatesToMaster(self);
			int tag=0, end_tag=0;
			MPI_Irecv(&(end_tag),1,MPI_UNSIGNED,0,tag,MPI_COMM_WORLD,&(req_end_tag));
			req_object=getGlobalUpdatesFromMaster(self);
		} else{ // poll for receive
			MPI_Test(&req_end_tag, &completedFlag, &stat);
		}

	}
}


void saveFiles(MasterClass* self, int iter){
	stringstream ss;
	ss<<iter;
	string temp_file = self->filenameSave;
	string suffix="_";
	suffix.append(ss.str());
	temp_file.append(suffix);
	self->pi.save(temp_file.append("_PI.csv"), csv_ascii);
	self->phi.save(temp_file.append("_PHI.csv"), csv_ascii);
	self->ll.save(temp_file.append("_LL.csv"), csv_ascii);
	cout<<"files saved"<<endl;
}


int getMasterParameters(MasterClass* self){
	// TO BE NOTED send sampling probabilities only in the order they are expected.

	// Gibbs Sampling
	// At present going for fixed number of iterations

	self->pi = zeros<fmat>(self->num_users,self->K);
	self->phi = zeros<fmat>(self->K,self->vocab_size);
	cout<<"Begining of getMasterParameters"<<endl;
	int inner_iter = 0;
	for(inner_iter=0;  inner_iter<self->sampleIters; inner_iter++){  //start from 0
		receiveAndSyncFromSlaves(self);

		//sum_Nkk, sum_Sum_kk1, sum_Sum_kk2, sum_Nuk, sum_Nkw = self.getTheSums()
		umat sum_Nkk = getTheSum(self->Nkk);
		self->ll(inner_iter)=calculate_joint_log_likelihood(self);
		cout<<inner_iter<<endl;
		cout<< self->ll(inner_iter)<<endl;
		if(inner_iter%100==0)
			saveFiles(self, inner_iter);
		sendGlobalupdatesToSlaves(self,0);

	}

	fmat temp_pi=zeros<fmat>(self->num_users,self->K);//np.zeros((self.num_users,self.K));
	fmat temp_phi=zeros<fmat>(self->K,self->vocab_size);
	for(int iter=0; iter< self->outer; iter++){
		for(int inner=0; inner<self->L; inner++){
			receiveAndSyncFromSlaves(self);
			sendGlobalupdatesToSlaves(self,0);
		}

		cout<< iter<<endl;

		//		cout<<self->ll.n_elem<<" "<<self->ll.n_cols<<" "<<inner_iter<<endl;
		//		cout<<self->ll<<endl;
		self->ll(inner_iter)=calculate_joint_log_likelihood(self);
		cout<< self->ll(inner_iter)<<endl;

		inner_iter = inner_iter+1;
		if(inner_iter%100==0)
			saveFiles(self, inner_iter);
		// estimate Pi
		umat sum_Nuk=getTheSum(self->Nuk);
		temp_pi = sum_Nuk+repmat(self->alpha,self->num_users,1);
		//		for(int u=0; u<self->num_users; u++){
		//			for(int k=0; k<self->K; k++){
		//				temp_pi(u,k)=self->Nuk(u,k)+self->alpha(k);
		//			}
		//		}
		//TODO: write matrix repmat and reshape operations or use some other array trick
		//		temp_pi=temp_pi/np.tile(np.sum(temp_pi,1).reshape(temp_pi.shape[0],1),(1,self.K));
		temp_pi = 1.0*temp_pi/repmat(sum(temp_pi,1),1,self->K);

		self->pi=self->pi+temp_pi;

		//TODO: estimate phi
		fmat sum_Nkw =  conv_to<fmat>::from(getTheSum(self->Nkw));
		fmat temp_mat(self->K, self->vocab_size);
		temp_phi =sum_Nkw + temp_mat.fill(self->beta); //repmat(self->beta,self->K, self->vocab_size);
		temp_phi = temp_phi*1.0/repmat(sum(temp_phi,1),1,self->vocab_size);

		self->phi = self->phi+temp_phi;
		//		for k in xrange(0,self.K):
		//phi[k,:]=phi[k,:]+((sum_Nkw[k,:]+self.beta)/(np.sum(sum_Nkw[k,:])+self.beta*self.vocab_size))
	}
	//TODO: send the end msg for the slaves
	// stop the slaves as computation is finished
	sendGlobalupdatesToSlaves(self,-1);
	cout<<"helloiamhere\n";
	self->pi=1.0*self->pi/self->outer;
	self->phi=1.0*self->phi/self->outer;
	cout<<"pi"<<self->pi<<endl;
	cout<<"phi"<<self->phi<<endl;

//	string temp_file = self->filenameSave;
//	self->pi.save(temp_file.append("_PI.csv"), csv_ascii);
//	self->phi.save(temp_file.append("_PHI.csv"), csv_ascii);
	cout<< inner_iter;
	saveFiles(self, inner_iter);
//	self->ll.save(temp_file.append("_LL.csv"), csv_ascii);
	return 1;
}

void slaveSampler(SlaveClass* self){ //TODO: change sum(suff stats), write to local Nuk and read from global_Nuk
	cout<< "In slaveSampler, rank "<< self->rank<<endl;
	int start_index = (self->rank-1)*self->slave_width;
	int end_index = (self->rank)*self->slave_width;
	int user_p_orig,cumul_count, word_count;
	umat temp_mat;

	for(int user_p=0;user_p<self->slave_width; user_p++){
		user_p_orig = user_p + start_index;
		cout<<user_p_orig<<endl;
		for(int user_q=0; user_q<self->num_users; user_q++){
			if(user_p_orig==user_q)       //if not then we will be sampling p->q twice. Wrong : y_pq != Y_qp
				continue;

			//get current assignment of k to p->q and q->p
			int p_k = self->Zuu1(user_p,user_q);
			int q_k = self->Zuu2(user_q,user_p);

			int y_pq1 = self->Y1(user_p,user_q);  //point to note: y is 0,1, we don't need 1,2
			int y_pq2 = self->Y2(user_p,user_q);
			temp_mat = self->Nuk_p+self->Nuk;
			fmat multi_probs = calculate_Mult_Probs_mmsbZ(
					self,
					temp_mat.row(user_p_orig),     // get counts from overall stats
					temp_mat.row(user_q),
					y_pq1,
					y_pq2,
					p_k,
					q_k
			);

			//decrement current counts
			self->Nkk[p_k,q_k] = self->Nkk[p_k,q_k] - 1; //decrement Ngh,y_pq
			self->Sum_kk1(p_k,q_k) = self->Sum_kk1(p_k,q_k)-y_pq1;
			self->Sum_kk2(p_k,q_k) = self->Sum_kk2(p_k,q_k)-y_pq2;
			self->Nuk(user_p_orig,p_k) = self->Nuk(user_p_orig,p_k) - 1;
			self->Nuk(user_q,q_k) = self->Nuk(user_q,q_k) - 1;

			// sample Z_pq and Z_qp together
			//                print multi_probs
			Mat<s16> p_kq_k = multinomial_bivariate_sample(self, multi_probs);
			p_k = p_kq_k(0);
			q_k = p_kq_k(1);
			self->Zuu1(user_p,user_q) = p_k;
			self->Zuu2(user_q,user_p) = q_k;

			self->Nkk(p_k,q_k) = self->Nkk(p_k,q_k) + 1; // increment Ngh,y_pq
			self->Sum_kk1(p_k,q_k) = self->Sum_kk1(p_k,q_k)+y_pq1;
			self->Sum_kk2(p_k,q_k) = self->Sum_kk2(p_k,q_k)+y_pq2;
			self->Nuk(user_p_orig,p_k) = self->Nuk(user_p_orig,p_k) + 1;
			self->Nuk(user_q,q_k) = self->Nuk(user_q,q_k) + 1;
		}
		// sample for LDA
		cumul_count = 0;
		for(int word=0; word<self->vocab_size; word++){
			word_count = self->W(user_p,word);
			if (word_count==0)
				continue;
			for(int word_num=cumul_count; word_num<cumul_count+word_count; word_num++){
				int p_k = self->Zuw[user_p](word_num);

				if (p_k<0)
					cout<<"why is sampled topic -ve?"<<endl;
				// read from global
				temp_mat = self->Nuk_p+self->Nuk;
				fmat mult_probs=calculate_Mult_Probs_ldaZ(self, temp_mat.row(user_p_orig), p_k, word);       // holy mother why did u use user_p

				self->Nkw(p_k,word) = self->Nkw(p_k,word) -1;
				self->Nuk(user_p_orig,p_k) = self->Nuk(user_p_orig,p_k) -1;

				p_k = lda_multinomial_sample(self,mult_probs);

				self->Zuw[user_p](word_num) = p_k;    // word of Z and Nkw must not be the same
				self->Nkw(p_k,word)= self->Nkw(p_k,word)+ 1;
				self->Nuk(user_p_orig,p_k) = self->Nuk(user_p_orig,p_k) + 1;     // Zs from MMSB and LDA are piling up
			}
			cumul_count = cumul_count + word_count;
		}
		//            except Exception, e:
		//                print traceback.format_exc()
		//                print 'Fuced in index slaveSampler'
		//                print (self.Nuk_p+self.Nuk)[user_p_orig,:], (self.Nuk_p+self.Nuk)[user_q,:], y_pq1, y_pq2, p_k, q_k
		//                print "sum: Nuk, Nkk, Nkw", np.sum((self.Nuk_p+self.Nuk)), np.sum((self.Nuk_p+self.Nuk)), np.sum((self.Nuk_p+self.Nuk))
		//                print "mult_probs", multi_probs
		//                print 'In debugger\n'#, self.Nuk
		//                raise Exception()
	}
	cout<<"SLAVE: At the end of slaveSampler, rank "<< self->rank<<endl;
}


fmat calculate_Mult_Probs_mmsbZ(SlaveClass* self,umat Npk, umat Nqk,int y_pq1, int y_pq2, int curr_p, int curr_q){
	//TODO: read from global
	fmat multi_probs= zeros<fmat>(self->K*self->K,1);
	umat temp_mat;
	int indx=0;
	for(int p_k=0;p_k<self->K;p_k++){
		for(int q_k=0;q_k<self->K;q_k++){
			temp_mat = self->Nkk_p+self->Nkk;
			int n_kk = temp_mat(p_k,q_k);
			int alpha_p=self->alpha(p_k);
			int alpha_q=self->alpha(q_k);
			temp_mat = self->Sum_kk1_p+self->Sum_kk1;
			int sum_local1 = temp_mat(p_k,q_k);
			int sum_local2 = temp_mat(p_k,q_k);
			if( p_k==curr_p && q_k==curr_q){
				n_kk = n_kk-1;
				sum_local1 = sum_local1 - y_pq1;
				sum_local2 = sum_local2 - y_pq2;
				alpha_p = alpha_p-1;
				alpha_q = alpha_q-1;
			}
			float p=1.0/((1.0/self->eta)+n_kk+1);
			p=1-p;
			sum_local1 = sum_local1 + self->kappa;
			sum_local2 = sum_local2 + self->kappa;
			float block_part1 = log_nbinpdf(y_pq1,sum_local1,p);        // nbinpdf returns zero if y_pq is not and integer
			float block_part2 = log_nbinpdf(y_pq2,sum_local2,p);
			float log_probs = block_part1 + block_part2;
			log_probs += log(((Npk(p_k)+alpha_p))) + log(((Nqk(q_k)+alpha_q)));
			if(log_probs < C_LOG_LIMIT)
				log_probs = C_LOG_LIMIT;
			float probs = exp(log_probs);
			multi_probs[indx]= probs;
			if (probs==0)
				cout<<y_pq1<<" "<<sum_local1<<" "<<y_pq2<<" "<<sum_local2<<" "<<p<<" "<<log_probs<<" "<<block_part1<<" "<<block_part2<<endl;
			indx=indx+1;
		}
	}
	//        except:
	//            print traceback.format_exc()
	//            print 'Fuced in index calculate_Mult_Probs_mmsbZ'
	//            print Npk , Nqk, q_k, p_k, curr_p, curr_q
	//            print "sum: Nuk, Nkk, Nkw", np.sum(self.Nuk_p+self.Nuk), np.sum(self.Nkk_p+self.Nkk), np.sum(self.Nkw_p+self.Nkw)
	//            raise Exception()
	//#            print 'In debugger\n', self.Nuk
	//        if np.sum(multi_probs)==0:
	//            print multi_probs
	//            print Npk, Nqk, curr_p, curr_q, y_pq1, y_pq2
	//            print "sum: Nuk, Nkk, Nkw", np.sum(self.Nuk_p+self.Nuk), np.sum(self.Nkk_p+self.Nkk), np.sum(self.Nkw_p+self.Nkw)
	//            raise Exception("Dude wtf?")
	return multi_probs;
}

float log_nbinpdf(int k, int r, float pr){
	return lgamma(r + k) - lgamma(r) - lgamma(k+1) + r*log(pr)+k*log(1-pr);
}

fmat calculate_Mult_Probs_ldaZ(SlaveClass* self, umat Npk, int curr_p, int word){
//        TODO change the count reads fromglobal sum
	umat temp_mat;
	fmat multi_probs= zeros<fmat>(self->K*self->K,1);
        int indx=0;
        for(int p_k=0;p_k<self->K; p_k++){
            float alpha=self->alpha(1);
            float beta = self->beta;
            float sum_beta = self->beta*self->vocab_size;    // this should ideally be sum of betas but we are takign uniform betas
            if (p_k==curr_p){
                alpha = alpha -1;
                beta = beta -1;
                sum_beta = sum_beta -1;
            }
            temp_mat = self->Nkw_p+self->Nkw;
            multi_probs(indx) = (temp_mat(p_k,word)+beta)/(accu(temp_mat.row(p_k))+sum_beta);
            multi_probs(indx) = multi_probs(indx) * (Npk(p_k)+alpha);
            indx=indx+1;
        }
        return multi_probs;
}



fmat vectorizeFloat(fmat input, float (*function)(float)){
	fmat result(1,input.n_elem);
	for(int i=0;i<input.n_elem;i++){
		result(i)=(* function)(input(i));
	}
	return result;
}

fmat gammaln(fmat input){
	return vectorizeFloat(input, boost::math::lgamma);
}



float calculate_joint_log_likelihood(MasterClass* self){ //TODO: sum(suff stats)
    float ll=0;
    // Nkk
    fmat sum_Nkk = conv_to<fmat>::from(getTheSum(self->Nkk));
    fmat sum_Nuk = conv_to<fmat>::from(getTheSum(self->Nuk));
    fmat sum_Nkw = conv_to<fmat>::from(getTheSum(self->Nkw));
    fmat sum_Sum_kk1 = conv_to<fmat>::from(getTheSum(self->Sum_kk1));
    fmat sum_Sum_kk2 = conv_to<fmat>::from(getTheSum(self->Sum_kk2));

    for( int g=0; g<self->K; g++){
        for(int h=0;h<self->K; h++){
            float sumLnGammaY_pq1 = getSumLnGammaY_pq(self, g, h, self->Y1);
            float sumLnGammaY_pq2 = getSumLnGammaY_pq(self, g, h, self->Y2);
        //         Sum_kk(g,h)
//                if sum_gh1~=Sum_kk1(g,h)
//                    'screwed badly'
//                    return;

            ll=ll + lgamma(sum_Sum_kk1(g,h)+self->kappa) -(sum_Sum_kk1(g,h)+self->kappa)*(sum_Nkk(g,h)+1/self->eta)-self->kappa*log(self->eta)-lgamma(self->kappa); //- sumSigma gammaln(Y_pq+1)
            ll = ll-sumLnGammaY_pq1;

            ll = ll + lgamma(sum_Sum_kk2(g,h)+self->kappa) -(sum_Sum_kk2(g,h)+self->kappa)*(sum_Nkk(g,h)+1/self->eta);
            ll = ll-sumLnGammaY_pq2;
        }
    }
    // likelihood from Z part
//    float temp1 = np.sum(gammaln(sum_Nuk+np.tile(self.alpha,(self.num_users,1)))-gammaln(np.tile(self.alpha,(self.num_users,1))));
    fmat temp_mat2=repmat(self->alpha,self->num_users,1);
//    temp_mat1 += conv_to<fmat>::from(sum_Nuk);
    float temp1 = accu(gammaln(sum_Nuk+temp_mat2)-gammaln(temp_mat2));
//    temp1 = temp1 + np.sum(gammaln(np.tile(np.sum(self.alpha),(1,self.num_users))) - gammaln(np.sum(sum_Nuk, axis=1)+np.tile(np.sum(self.alpha),(1,self.num_users))))
    temp1 = temp1 + accu(gammaln(sum(temp_mat2,1)) - gammaln(sum(sum_Nuk+temp_mat2,1)));
//        print "Nuk", temp,temp1
    ll=ll+temp1;
    //LDA word component
//    temp1 = np.sum(gammaln(sum_Nkw+np.tile(self.beta,(self.K,self.vocab_size))))
    fmat temp_mat1(self->K,self->vocab_size); //repmat(self->beta,self->K,self->vocab_size);
    temp_mat1.fill(self->beta);
    temp1 = accu(gammaln(sum_Nkw+temp_mat1));
//    temp1 = temp1 + np.sum(-gammaln(np.sum(sum_Nkw, axis=1)+np.tile(self.vocab_size*self.beta,(1,self.K))))
    fmat temp_mat(self->K,1) ;//repmat(self->vocab_size*self->beta,self->K,1);
    temp_mat.fill(self->vocab_size*self->beta);
    temp1 = temp1 + accu(-gammaln(sum(sum_Nkw, 1)+temp_mat));
//	cout<<"Nkw "<<"temp1 "<<temp1<<endl;
    ll = ll+temp1;
    return ll;
}


float getSumLnGammaY_pq(MasterClass* self, int g, int h, umat Y){
    s16mat Z1 = self->Zuu1;
    s16mat Z2 = self->Zuu2;

//    r1,c1 = np.nonzero(Z1==g);
    uvec colVec = find(Z1==g);
    uvec cols = floor(colVec/Z1.n_cols);
    uvec rows = colVec-cols*Z1.n_cols;
    uvec newColVec = rows*Z1.n_cols+cols;
//    Z2_2 = Z2[c1,r1]; // p->q become q->p
    uvec Z2_2 = conv_to<uvec>::from(Z2.elem(newColVec));
    uvec ColVec2 = find(Z2_2==h);
    uvec finalColVec = conv_to<uvec>::from(colVec.elem(ColVec2));

//    r = r1[Z2_2==h]
//    c = c1[Z2_2==h]
    fmat y_gh = conv_to<fmat>::from(Y.elem(finalColVec));

//    y_gh = Y[r,c];
//        sum_gh=np.sum(y_gh)
//        print y_gh+1
    int sumLnGammaY_pq = accu(gammaln(y_gh+1));
    return sumLnGammaY_pq;
}


Mat<s16> multinomial_bivariate_sample(VariableClass* self, fmat mult_probs){
	mult_probs = mult_probs/accu(mult_probs);
	mult_probs = cumsum(mult_probs,0);	// cum sum along the columns

//	cout<<mult_probs<<endl; //<< cumsum(mult_probs,1)<<endl;
	float rand_generated = randomGen();
	Mat<s16> k_pk_q(1,2);
	//        index = np.nonzero(mult_probs>np.random.random())[0][0];
	uvec indices = find(mult_probs>=rand_generated); //TODO: there shudnt be equality
	int index = indices(0);
	k_pk_q(0) = ceil(index/self->K);
	k_pk_q(1) = index%self->K;

	//        if k_q==0:    # this is not valid for python
	//        k_q=self.K
//	cout<<k_pk_q<<endl;
	return k_pk_q;
}

int lda_multinomial_sample(SlaveClass* self,fmat mult_probs){
	mult_probs = mult_probs/accu(mult_probs);
	mult_probs = cumsum(mult_probs,0);	// cum sum along the columns;
	float rand_generated = randomGen();
	uvec indices = find(mult_probs>=rand_generated); //TODO: there shudnt be equality
//	if(indices.n_elem==0)
//		cout<<"mult_probs, rand_generated, indices, rank"<<mult_probs<<" "<<" "<<rand_generated<<" "<< indices<<" "<<self->rank<<endl;
	int index = indices(0);
    return index;
}

int main(int argc, char** argv){
	MPI_Init(&argc, &argv);
//    fmat a;//sio.loadmat("monkLK.mat", {})
//    string filename = "abhimanu_sp_matrix_V2_5000_binary.csv";
//    a.load(filename, csv_ascii);
//    a=a.submat(0,0,999,999);
//    print a['y']
//    cls = MMSB(a['y'], 3, 1000, 100, 2, 'test_monk_python')

    VariableClass *self ;//= new VariableClass();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &(rank));
    if (rank==MASTER_RANK)
    	self = new MasterClass();
    else
    	self = new SlaveClass();

    string filenamesRead[] = {"edge3.csv", "edge1.csv", "user_texts_ok.csv"};
    string filenameSave = "cancer-dataset";
//    initialize(VariableClass* self, string* filenamesRead, int K, int sampleIters, int outer, int L, int assigned_users, string filenameSave)
    initialize(self, filenamesRead, 3, 3000, 500, 5, 10, filenameSave);
//    initialize(self, a, 3, 3000, 500, 5, filename);
//    cout<<"cls num_user "<<self->num_users<<endl;
//	cout<< "Nkk sum "<<accu(self->Nkk)<<endl;//np.sum(self.Nkk);
//	cout << "Nuk sum " <<accu(self->Nuk)<<endl; //np.sum(self.Nuk);
//	cout<<"done filling initial values"<<endl;
//	cout<<"num_user "<<self->num_users<<endl;
//	cout<<"Nuk "<< self->Nuk<<endl;
//	cout<<"Zuu "<< self->Zuu<<endl;
//    getParameters(self);
    runComputation(self);
}
