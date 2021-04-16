#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <snp_model.hpp> // "../include/snp_model.hpp" // <snp_model.hpp>
#include <snp_static.hpp>
#include <math.h>
#include <iostream>

// Algorithms
#define CPU      		0
#define GPU_CUBLAS 		1
#define GPU_CUSPARSE 	2
#define GPU_SPARSEREP 	3

template <typename T>
void printMatx(T trans_matrix, int n, int m){
	for (int i=0; i<m; i++){
		
		for (int j=0; j<n; j++){
			std::cout << trans_matrix[i*n + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
	
}

void testSNP_gpu(){
//Loading one SNP model
	int m = 5; //num reglas
	int n = 3; //num neuronas

	SNP_static_ell TestModel(n, m);
	int C0[3] = {2,1,1};
	for (int i=0; i<n; i++){
		TestModel.set_spikes (i, C0[i]);
	}

	//add_rule (uint nid, uchar e_n, uchar e_i, uchar c, uchar p) 
	TestModel.add_rule(0, 2, 1, 1, 1);
	TestModel.add_rule(0, 2, 1, 2, 1);
	TestModel.add_rule(1, 1, 1, 1, 1);
	TestModel.add_rule(2, 1, 1, 1, 1);
	TestModel.add_rule(2, 2, 1, 2, 0);

	printMatx<uint *>(TestModel.rule_index,n+1,1);

	TestModel.add_synapse(0,1);
	TestModel.add_synapse(1,0);
	TestModel.add_synapse(0,2);
	TestModel.add_synapse(1,2);
	TestModel.add_synapse(2,n);

	printMatx<short *>(TestModel.conf_vector,n,1);
	printMatx<short *>(TestModel.trans_matrix, n, m);
	TestModel.transition_step(); 
	TestModel.load_to_cpu ();

	// printMatx<ushort *>(TestModel.spiking_vector,m,1);
	printMatx<short *>(TestModel.conf_vector,n,1);
}

void testSNP_cpu(){
//Loading one SNP model
	int m = 5; //num reglas
	int n = 3; //num neuronas

	SNP_static_cpu TestModel(n, m);
	int C0[3] = {2,1,1};
	for (int i=0; i<n; i++){
		TestModel.set_spikes (i, C0[i]);
	}

	//add_rule (uint nid, uchar e_n, uchar e_i, uchar c, uchar p) 
	TestModel.add_rule(0, 2, 1, 1, 1);
	TestModel.add_rule(0, 2, 1, 2, 1);
	TestModel.add_rule(1, 1, 1, 1, 1);
	TestModel.add_rule(2, 1, 1, 1, 1);
	TestModel.add_rule(2, 2, 1, 2, 0);

	printMatx<uint *>(TestModel.rule_index,n+1,1);

	TestModel.add_synapse(0,1);
	TestModel.add_synapse(1,0);
	TestModel.add_synapse(0,2);
	TestModel.add_synapse(1,2);
	TestModel.add_synapse(2,n);

	printMatx<ushort *>(TestModel.conf_vector,n,1);
	printMatx<short *>(TestModel.trans_matrix, n, m);
	
	TestModel.transition_step(); 
	printMatx<ushort *>(TestModel.spiking_vector,m,1);
	printMatx<ushort *>(TestModel.conf_vector,n,1);
}

int main(int argc, char* argv[])
{
	//////////////////////
	
	testSNP_gpu();
	// testSNP_cpu();


	/////////////////////
	
	int n = 12;
	char* file=NULL;
	int algorithm = CPU;
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
		
	if (argc>1) {
		file = argv[1];	
	}
	
	if (argc>2) {
		algorithm = atoi(argv[2])-1;
	}
	// Read the input file

	
		
	switch (algorithm)
	{
		case CPU:
			
			//init_params(MAP1,n,0.15,DEBUG,algorithm,&params);
			//init_vars(8,10,&params,&vars);
		break;
		case GPU_CUBLAS:
			//init_params(MAP2,n,0.15,DEBUG,algorithm,&params);
			//init_vars(32,9.3,&params,&vars);
		break;
		case GPU_CUSPARSE:
			//init_params(MAP3,n,0.15,DEBUG,algorithm,&params);
			//init_vars(21.5,21.5,&params,&vars);
		break;		
		case GPU_SPARSEREP:
			//init_params(MAP3,n,0.15,DEBUG,algorithm,&params);
			//init_vars(21.5,21.5,&params,&vars);
		break;		
		default:
			printf("Invalid algorithm\n");
			return 0;
	}
	
	
	
	//params.debug=1;	
	//while (!vars.halt) {
	//snp_simulator(&params,&vars);
	//}
		
	//free_memory(&params,&vars);
	
	return 0;
}
