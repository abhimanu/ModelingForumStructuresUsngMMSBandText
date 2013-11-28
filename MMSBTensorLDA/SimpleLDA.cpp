/*
 * SimpleLDA.cpp
 *
 *  Created on: Apr 9, 2013
 *      Author: abhimank
 */

#include "SimpleLDA.h"
#include <string>

SimpleLDA::SimpleLDA() {
	// TODO Auto-generated constructor stub

}

SimpleLDA::~SimpleLDA() {
	// TODO Auto-generated destructor stub
}

void SimpleLDA::initilaize(string* filenamesRead, int K, int sampleIters, int outer, int L, int assigned_users, string filenameSave) {
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
}
