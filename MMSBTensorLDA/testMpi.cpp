/*
 * testMpii.cpp
 *
 *  Created on: Feb 25, 2013
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
	MPI_Request r;
	int msg_size=MSGSIZE;
	umat msg = ones<umat>(msg_size,msg_size);
	for(int i=0; i<10; i++){
		msg = initMsg(msg)*(i+1);
		cout << "FULL MSG"<<msg<<endl;
//		int msg = i*2;
		umat sub_msg = msg.rows(0,msg_size/2-1); //include bot sides of ranges
		MPI_Isend(&(sub_msg(0)), sub_msg.n_elem, MPI_UNSIGNED, 1, 0, MPI_COMM_WORLD,&r);
		cout<<"sent msg for rank 1\n"<< sub_msg<<endl;
		msg+=1;
		sub_msg = msg.rows(msg_size/2, msg_size-1);
		MPI_Isend(&(sub_msg(0)), sub_msg.n_elem, MPI_UNSIGNED, 2, 0, MPI_COMM_WORLD,&r);
		cout<<"sent msg for rank 2\n"<< sub_msg<<endl;
		sleep(2);
	}
	int completedFlag;
	MPI_Status status;
	MPI_Test(&r, &completedFlag, &status);
	while (!completedFlag){
		cout<< "still waiting for the last ack, rank"<< rank << endl;
		MPI_Test(&r, &completedFlag, &status);
		sleep(1);
	}
	cout<<status.MPI_SOURCE<<status._count<<endl;
	MPI_Finalize();
}


void slaveFunction(int rank){
	MPI_Request r;
	MPI_Status status;
	int msg_size=MSGSIZE;
	umat msg = zeros<umat>(msg_size/2,msg_size);
	int completedFlag=0;
	for(int i=0; i<10; i++){
		MPI_Irecv(&(msg(0)), msg.n_elem, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &r);
		MPI_Test(&r, &completedFlag, &status);
		while(!completedFlag){
			cout<< "Waiting in process rank "<<rank<<" for the mssg# "<<i<< endl;
			MPI_Test(&r, &completedFlag, &status);
			sleep(1);
		}
		cout<<"Received Msg"<<" in rank "<<rank<<"\n"<<msg<<endl;
	}
	MPI_Finalize();
}

int main(int argc, char** argv){
	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm comm;
	if(rank==0)
		masterFunction(rank);
	else
		slaveFunction(rank);
}
