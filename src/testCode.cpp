#include<iostream>
#include<stdlib.h>

using namespace std;
int main(int argc, char** argv){
	for(int i=0; i<50; i++)
		cout <<rand()*1.0/RAND_MAX<<", "<<RAND_MAX<<endl;
    return 0;
}
