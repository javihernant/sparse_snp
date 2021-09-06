#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <snp_model.hpp> // "../include/snp_model.hpp" // <snp_model.hpp>
#include <snp_static.hpp>
#include <math.h>
#include <iostream>
#include <unistd.h>

       

// Algorithms
#define CPU      		0
#define GPU_SPARSE		1
#define GPU_ELL 		2
#define GPU_OPTIMIZED	3
#define GPU_CUBLAS 		4
#define GPU_CUSPARSE 	5




// void testSNP_cpu(){
// //Loading one SNP model
// 	int m = 5; //num reglas
// 	int n = 3; //num neuronas

// 	SNP_static_cpu TestModel(n, m);
// 	int C0[3] = {2,1,1};
// 	for (int i=0; i<n; i++){
// 		TestModel.set_spikes (i, C0[i]);
// 	}

// 	//add_rule (uint nid, uchar e_n, uchar e_i, uchar c, uchar p) 
// 	TestModel.add_rule(0, 2, 1, 1, 1);
// 	TestModel.add_rule(0, 2, 1, 2, 1);
// 	TestModel.add_rule(1, 1, 1, 1, 1);
// 	TestModel.add_rule(2, 1, 1, 1, 1);
// 	TestModel.add_rule(2, 2, 1, 2, 0);

// 	printMatx<uint *>(TestModel.rule_index,n+1,1);

// 	TestModel.add_synapse(0,1);
// 	TestModel.add_synapse(1,0);
// 	TestModel.add_synapse(0,2);
// 	TestModel.add_synapse(1,2);
// 	TestModel.add_synapse(2,n);

// 	printMatx<ushort *>(TestModel.conf_vector,n,1);
// 	printMatx<short *>(TestModel.trans_matrix, n, m);
	
// 	TestModel.transition_step(); 
// 	printMatx<ushort *>(TestModel.spiking_vector,m,1);
// 	printMatx<ushort *>(TestModel.conf_vector,n,1);
// }

// void testSNP_gpu(){
// 	//TODO: Add one output neuron
// 	//Loading one SNP model
// 	int m = 5; //num reglas
// 	int n = 3; //num neuronas

// 	SNP_static_optimized TestModel(n, m);
// 	int C0[3] = {2,1,1};
// 	for (int i=0; i<n; i++){
// 		TestModel.set_spikes (i, C0[i]);
// 	}

// 	//add_rule (uint nid, uchar e_n, uchar e_i, uchar c, uchar p) 
// 	TestModel.add_rule(0, 2, 1, 1, 1);
// 	TestModel.add_rule(0, 2, 1, 2, 1);
// 	TestModel.add_rule(1, 1, 1, 1, 1);
// 	TestModel.add_rule(2, 1, 1, 1, 1);
// 	TestModel.add_rule(2, 2, 1, 2, 0);

// 	printMatx<uint *>(TestModel.rule_index,n+1,1);

// 	TestModel.add_synapse(0,1);
// 	TestModel.add_synapse(1,0);
// 	TestModel.add_synapse(0,2);
// 	TestModel.add_synapse(1,2);
// 	TestModel.add_synapse(2,n);

// 	//////////////////////////////////////////////////
// 	for(int nid=0; nid<n; nid++){
// 		printf("%d ", TestModel.get_spikes(nid));
// 	}
// 	printf("\n");
// 	///////////////////////////////////////////////////

// 	printMatx<short *>(TestModel.trans_matrix, n, m);
	
// 	TestModel.transition_step(); 
	
// 	// printMatx<ushort *>(TestModel.spiking_vector,m,1);
	
// 	//////////////////////////////////////////////////
// 	for(int nid=0; nid<n; nid++){
// 		printf("%d ", TestModel.get_spikes(nid));
// 	}
// 	printf("\n");
// 	///////////////////////////////////////////////////
	
// }

void testOrdenarNums(int* nums, int size, int verbosity, bool write2csv){
	int n= size*3; //number of neurons is number of numbers * 3 layers. 
	int m = size + size*size; //each neuron in the first layer has one rule. Each neuron in the second layer has size (of the array of nums to be sorted) rules. There are "size" neurons in each layer (input, second, output).

	SNP_static_optimized TestModel(n, m, GPU_OPTIMIZED, verbosity, write2csv);
	//set spikes of neurons in first layer and add their rules
	for(int i=0; i<size; i++){
		TestModel.set_spikes (i, nums[i]);
		//add_rule (uint nid, uchar e_n, uchar e_i, uchar c, uchar p) 
		TestModel.add_rule(i, 1, 0, 1, 1,0);	
	}

	int e_n_aux = size;
	//add rules in neurons of 2nd layer
	for(int j=size; j<size*2; j++){
		for(int e_n=size; e_n>=1; e_n--){
			if(e_n == e_n_aux){
				TestModel.add_rule(j, e_n, 1, e_n, 1,0);
			}else{
				TestModel.add_rule(j, e_n, 1, e_n, 0,0);
			}
		}
		e_n_aux--;
	}

	//Connect 1st 2nd and 3rd layers
	for(int i=0; i<size; i++){
		for(int j=size; j<size*2; j++){
			TestModel.add_synapse(i,j);
		}
	}

	e_n_aux = size;
	for(int j=size; j<size*2; j++){
		for(int offset=0; offset<e_n_aux; offset++){
			TestModel.add_synapse(j,j+size+offset);

		}
		e_n_aux--;
	
	}
	
	// printf("Initial conf_vector: ");
	// //////////////////////////////////////////////////
	// for(int nid=0; nid<n; nid++){
	// 	printf("%d ", TestModel.get_spikes(nid));
	// }
	// printf("\n\n");
	// ///////////////////////////////////////////////////
	
	TestModel.compute(500); 
	

}

// void testOrdenarNumsELL(){
	
// }

// void testOrdenarNumsOptimized(){
	
// }

void testDelays(int verbosity, int write2csv){
	
	//Loading one SNP model
	uint m = 5; //num reglas
	uint n = 3; //num neuronas
	
	SNP_static_optimized TestModel(n, m, GPU_OPTIMIZED, verbosity, write2csv);
	int C0[3] = {0,1,1};
	for (int i=0; i<n; i++){
		TestModel.set_spikes (i, C0[i]);
	}

	// printf("Initial conf_vector: ");

	// //////////////////////////////////////////////////
	// for(int nid=0; nid<n; nid++){
	// 	printf("%d ", TestModel.get_spikes(nid));
	// }
	// printf("\n----------------------------------\n");
	// ///////////////////////////////////////////////////

	//add_rule (uint nid, short e_n, short e_i, short c, short p, ushort d) 
	TestModel.add_rule(0, 1, 1, 1, 1,0);
	TestModel.add_rule(0, 2, 1, 2, 0,0);
	TestModel.add_rule(1, 1, 1, 1, 1,0);
	TestModel.add_rule(1, 1, 1, 1, 1, 1);
	TestModel.add_rule(2, 1, 1, 1, 1,2);


	TestModel.add_synapse(0,1);
	TestModel.add_synapse(1,0);
	TestModel.add_synapse(0,2);
	TestModel.add_synapse(2,0);
	

	
	TestModel.compute(10);
	
	

}

void testSubsetSumNonUniformDelays(int S, int * v, int v_size, int verbosity, bool write2csv, int repetition){

	if(verbosity>=1){
		printf("test repetition #%d\n",repetition);
	}	
	int sum_of_v = 0;
	for(int i=0; i<v_size; i++){
		sum_of_v += v[i];
	}

	uint n = v_size*2 + sum_of_v +3; //num neuronas
	uint m = v_size*2*2 + sum_of_v + 2; //num reglas
	
	
	SNP_static TestModel(n, m, GPU_SPARSE, verbosity, write2csv, repetition);

	for (int i=0; i<v_size+1; i++){
		TestModel.set_spikes (i, 1);
	}


	//Adding rules. add_rule (uint nid, short e_n, short e_i, short c, short p, ushort d)
	TestModel.add_rule(0, 1, 1, 1, 1,0);
	for (int i=1; i<=v_size; i++){
		
		if((std::rand() % 2)==0){
			TestModel.add_rule(i, 1, 1, 1, 1,0);
			TestModel.add_rule(i, 1, 1, 1, 1,1);
		}else{
			TestModel.add_rule(i, 1, 1, 1, 1,1);
			TestModel.add_rule(i, 1, 1, 1, 1,0);

		}
		
	}

	
	for (int i=v_size+1; i<=v_size*2; i++){
		
		TestModel.add_rule(i, 2, 1, 2, 1,0);
		TestModel.add_rule(i, 1, 1, 1, 0,0);
	}

	int neuron = v_size*2+1;
	
	for (int i=0; i<v_size; i++){
		
		for(int offset=0; offset<v[i]; offset++){
			TestModel.add_rule(neuron+offset, 1, 1, 1, 1,0);
		}
		neuron+=v[i];
		
		
	}
	TestModel.add_rule(neuron, S, 1, S, 1,0);
	
	//Adding synapses

	for (int i=v_size+1; i<=v_size*2; i++){
		TestModel.add_synapse(0,i);
	}

	for (int i=1; i<=v_size; i++){
		TestModel.add_synapse(i,v_size*2+i);
	}

	int j_n = v_size*2 + 1;
	for (int i=0; i<v_size; i++){
		int i_n = v_size+1+ i;
		for(int c=0; c<v[i]; c++){
			
			TestModel.add_synapse(i_n,j_n);
			j_n++;
		}
			
	}

	//connecting to output neuron
	for(int i_n=v_size*2+1; i_n<j_n; i_n++){
		TestModel.add_synapse(i_n,j_n); 

	}
	//connecting out_neuron to enviroment neuron
	TestModel.add_synapse(j_n, j_n+1);
	

	TestModel.compute(4); //4 steps at most, 2 at minimum
	
	

}


int main(int argc, char* argv[])
{
	//////////////////////
	
	

	//verbosity
	//0 nada por pantalla
	//1 ultima configuracion
	//2 todas las configuraciones
	//lo anterior + spiking vectors, delays, trans_MX etc.

	int verbosity = 0;
	bool write2csv=false;
	int opt;

	while ((opt = getopt(argc, argv, "fv:")) != -1) {
		switch (opt) {
               case 'f':
                   write2csv = true;
                   break;
               case 'v':
                   verbosity = atoi(optarg);
                   break;
               default: /* '?' */
                   fprintf(stderr, "Usage: %s [-f] [-v verbositylevel] \n",
                           argv[0]);
                   exit(EXIT_FAILURE);
               }
           

	}
	if(write2csv){
		system("rm -r csv_solutions/*");
	}

	if(write2csv && verbosity==0){
		verbosity=1;
	}
	

	/////////////////Subset Sum//////////////////////
	// int numOfRepetitions = 100;
	// int v_size = 10;
	// int v[v_size];
	// int S=0;
	// int seed = 28;
	// std::srand(seed);
	// for(int i=0; i<v_size; i++){
		
	// 	v[i] = ( 1+ std::rand() % ( 10 + 1 ) ); //generates a number in the range 0-10


	// 	if((std::rand() % 100)<20){ //20% of the time the element will be chosen for the total sum (S)
	// 		S+=v[i];
	// 	}
	// }

	// for (int i=0; i<numOfRepetitions; i++){
	// 	testSubsetSumNonUniformDelays(S, v, v_size, verbosity, write2csv, i);
	// }    
	////////////////////////////////////////////////

	/////////////////Test Delays//////////////////////	
	
	// testDelays(verbosity,write2csv);

	//////////////////////////////////////////////////

	/////////////////Sorting numbers//////////////////////	
	int size = 100;
	int nums[size];
	for (int i=size; i>0; i--){
		nums[size-i]=i;
	}
	testOrdenarNums(nums,size, verbosity,write2csv);
	/////////////////////////////////////////////////////
	
	// int n = 12;
	// char* file=NULL;
	// int algorithm = CPU;
	//int seed = time(NULL);

	
	// if (argc==1) {
	// 	printf("SPARSESNP, a project to test sparse matrix representations of SNP on GPUs.\n");
	// 	printf("\nFormat: %s [i] [a]\n",argv[0]);
	// 	printf("Where: \n");
	// 	//printf("\n[i] is the input file describing the SNP to be simulated (check format in help file of repository)\n");
	// 	printf("\n[i] is the example index:\n");
	// 	printf("\t1 = sorting network\n");
	// 	printf("\t2 = \n");
	// 	printf("\n[a] is the algorithm index\n");
	// 	printf("\t1 = CPU (not implemented yet)\n");
	// 	printf("\t1 = GPU lineal algebra CUBLAS\n");
	// 	printf("\t2 = GPU sparse representation CUSPARSE\n");
	// 	printf("\t3 = GPU sparse representation\n");
	// 	return 0;
	// }
		
	// if (argc>1) {
	// 	file = argv[1];	
	// }
	
	// if (argc>2) {
	// 	algorithm = atoi(argv[2])-1;
	// }
	// Read the input file

	
		
	// switch (algorithm)
	// {
	// 	case CPU:
			
	// 		//init_params(MAP1,n,0.15,DEBUG,algorithm,&params);
	// 		//init_vars(8,10,&params,&vars);
	// 	break;
	// 	case GPU_CUBLAS:
	// 		//init_params(MAP2,n,0.15,DEBUG,algorithm,&params);
	// 		//init_vars(32,9.3,&params,&vars);
	// 	break;
	// 	case GPU_CUSPARSE:
	// 		//init_params(MAP3,n,0.15,DEBUG,algorithm,&params);
	// 		//init_vars(21.5,21.5,&params,&vars);
	// 	break;		
	// 	case GPU_SPARSEREP:
	// 		//init_params(MAP3,n,0.15,DEBUG,algorithm,&params);
	// 		//init_vars(21.5,21.5,&params,&vars);
	// 	break;		
	// 	default:
	// 		printf("Invalid algorithm\n");
	// 		return 0;
	// }
	
	
	
	//params.debug=1;	
	//while (!vars.halt) {
	//snp_simulator(&params,&vars);
	//}
		
	//free_memory(&params,&vars);
	
	return 0;
}
