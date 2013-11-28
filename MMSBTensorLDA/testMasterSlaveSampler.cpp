/*
 * testMasterSlaveSampler.cpp
 *
 *  Created on: Feb 26, 2013
 *      Author: abhimank
 */

#include<iostream>
#include<mpi.h>
#include "armadillo"

#define MSGSIZE 4;

using namespace arma;
using namespace std;

//TODO: pass buffers of type armadillo mat
//TODO: get the same kind of synchronization as in case of samplers and master



umat initMsg(umat msg){
	for(int i=0; i<msg.n_rows;i++)
		for(int j=0; j<msg.n_cols; j++)
			msg(i,j)=i*msg.n_cols+j;
	return msg;
}

void masterFunction(int rank){
	int msg_variety = 2;
	int msg_size=MSGSIZE;

	int num_procs;
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	int num_slaves = num_procs-1;
	int completedFlag;
	MPI_Status status[num_slaves];
	MPI_Request r[num_slaves][msg_variety];
	umat msgs[num_slaves];
	// init msgs to the slaves
//	cout<< "num_slaves in root "<<num_slaves<<endl;

	int init_msg[2]={num_slaves,msg_size/num_slaves};
//	cout<< "sizeof(init_msg[0]) in root "<<sizeof(init_msg[0])<<endl;
	for(int s=0;s<num_slaves;s++)	//blocking send
		MPI_Send(init_msg,sizeof(init_msg)/sizeof(init_msg[0]),MPI_INT,s+1,0,MPI_COMM_WORLD);
	cout<<"MASTER: Sent blocking init messages to slaves"<<endl;

	for(int s=0; s<num_slaves;s++){
		msgs[s] = zeros<umat>(msg_size/num_slaves,msg_size);
	}

//	cout<<"MASTER: "<< msgs[0]<<endl<<msgs[1] <<endl;
	// receives first local updates from the slaves
	for(int s=0;s<num_slaves;s++){	//blocking receive
		MPI_Recv(&(msgs[s][0]), msgs[s].n_elem, MPI_UNSIGNED, s+1, 0, MPI_COMM_WORLD, &(status[s]));
	}

	for(int s=0;s<num_slaves;s++)	//blocking receive
		cout<<"MASTER: first local slave counts received for rank "<<s+1<<"\n"<<msgs[s]<<endl;
	// compute global and send local; start for loop?

	// Send slaves all the global count except the end_tag msg to get them started
	for(int s=0;s<num_slaves;s++){
		MPI_Isend(&(msgs[s](0)), msgs[s].n_elem, MPI_UNSIGNED, s+1, 0, MPI_COMM_WORLD,&(r[s][1]));
		cout<<"MASTER: SENT msg for rank "<<s+1<<", to get it started computing\n"<< msgs[s]<<endl;
	}

	int end_tag =0 ;
	for(int i=0; i<10; i++){

		for(int s=0;s<num_slaves;s++)	//blocking receive
			MPI_Recv(&(msgs[s][0]), msgs[s].n_elem, MPI_UNSIGNED, s+1, 0, MPI_COMM_WORLD, &(status[s]));
		for(int s=0;s<num_slaves;s++)	//blocking receive
			cout<<"MASTER: RECVD from slave "<<s+1<<", loop # "<<i<<"\n"<<msgs[s]<<endl;

		for(int s=0;s<num_slaves;s++){
			msgs[s] = msgs[s] + 1;	// compute global
			msgs[s] = msgs[s] + 1;
		}
		//send complex msg = count + tag
		for(int s=0;s<num_slaves;s++){
			MPI_Isend(&(end_tag), 1, MPI_INT, s+1, 1, MPI_COMM_WORLD,&(r[s][0])); // change tags
			MPI_Isend(&(msgs[s](0)), msgs[s].n_elem, MPI_UNSIGNED, s+1, 0, MPI_COMM_WORLD,&(r[s][1]));
			cout<<"MASTER: SENT msg for rank "<<s+1<<", loop num "<<i<<"\n"<< msgs[s]<<endl;
		}
//		MPI_Isend(&(end_tag), 1, MPI_INT, 2, 1, MPI_COMM_WORLD,&r);
//		MPI_Isend(&(msgs[1](0)), msgs[1].n_elem, MPI_UNSIGNED, 2, 0, MPI_COMM_WORLD,&r);
//		cout<<"MASTER: SENT msg for rank 2, loop num "<<i<<"\n"<< msgs[1]<<endl;


//		sleep(2); // do we need this?
	}

	//send last msg = count + tag
	end_tag=-1; // change tags
	for(int s=0;s<num_slaves;s++){
		MPI_Isend(&(end_tag), 1, MPI_INT, s+1, 1, MPI_COMM_WORLD,&(r[s][0])); // change tags
		MPI_Isend(&(msgs[s](0)), msgs[s].n_elem, MPI_UNSIGNED, s+1, 0, MPI_COMM_WORLD,&(r[s][1]));
		cout<<"MASTER: SENT LAST msg for rank "<<s+1<<"\n"<< msgs[s]<<endl;
	}

	MPI_Test(r[1], &completedFlag, status);
	while (!completedFlag){
		cout<< "MASTER: still waiting for the last ack, rank"<< rank << endl;
		MPI_Test(r[1], &completedFlag, status);
		sleep(1);
	}
	cout<<status[0].MPI_SOURCE<<status[0]._count<<endl;
	MPI_Finalize();
}


void slaveFunction(int rank){
	MPI_Request r;
	MPI_Status status;
	int init_msg[2];
	int msg_size=MSGSIZE;
	int msg_variety = 2;
	// receive initial elements
	// blocking receive
//	cout<< "sizeof(init_msg[0]) in master "<<sizeof(init_msg[0])<<endl;
	MPI_Recv(init_msg, sizeof(init_msg)/sizeof(init_msg[0]), MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
	cout<<"SLAVES: Received first counts from master, rank"<<rank<<endl;
	// init local counts; need to have init_msgs, thus blocking
	umat msg = zeros<umat>(init_msg[1],msg_size);
	msg = initMsg(msg)+rank;
//	cout<<"SLAVES: "<<msg<<endl;
	int end_tag = 0;
	// Define a derived MPI data structure; We choose to do piecewise.

	//send these local updates; non-blocking send
	MPI_Isend(&msg(0), msg.n_elem, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &r);
	cout<<"SLAVES: Sent the first local updates, rank"<<rank<<endl;
	//Receive the first global updates; blocking recv since the slave cannt proceed in the begining
	// note that there are multiple sends from the root in the same block denoting diferent structs\
	// We do not ask for end_tag first time
//	MPI_Recv(&end_tag, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
	MPI_Recv(&(msg(0)), msg.n_elem, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &status);
	cout<<"SLAVES: Received first global msg from master, rank"<<rank<<"\n"<< msg <<endl;
	// send updated non-blocking local msg; and non-blocking recv; for loop?
//	receiveBlockingComplexStruct(&endTag, msg);
	MPI_Request req[msg_variety];
	MPI_Status stat[msg_variety];
	int completedFlag=0;
	int msg_num=0;
	while(end_tag>=0){ // all sends and recvs non-blocking
		// send first; since last was recv
		msg+=rank;
		MPI_Isend(&msg(0), msg.n_elem, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &r);
		// recv, and test_all** tricky
		cout<<"SLAVES: Sent the LOCAL msg # "<<msg_num<<" rank"<<rank<<endl<<msg<<endl;
		MPI_Irecv(&(end_tag), 1, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD, &req[1]);
		MPI_Irecv(&(msg(0)), msg.n_elem, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &req[1]);
//		MPI_Testall(msg_variety, req, &completedFlag, stat);
		MPI_Test(&req[1], &completedFlag, &stat[1]);
		while(!completedFlag){
			cout<< "SLAVES: Waiting in process rank "<<rank<<" for the msg# "<<msg_num<< endl;
//			MPI_Testall(2, req, &completedFlag, stat);
			MPI_Test(&req[1], &completedFlag, &stat[1]);
			sleep(1);
		}
		cout<<"SLAVES: Received GLOBAL Msg #"<< msg_num<<" in rank "<<rank<< ", status.MPI_ERROR"<<stat[0].MPI_ERROR<<"\n"<<msg<<endl;
		msg_num++;
	}
	MPI_Finalize();
}

int main(int argc, char** argv){
	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm comm;
	cout<< "my rank "<<rank<<endl;
	int num_procs;
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	cout<< "num_procs "<<num_procs<<endl;
	if(rank==0)
		masterFunction(rank);
	else
		slaveFunction(rank);
}
