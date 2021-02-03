#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <snp_model.hpp>
#include <math.h>

// Algorithms
#define CPU      		0
#define GPU_CUBLAS 		1
#define GPU_CUSPARSE 	2
#define GPU_SPARSEREP 	3

int main(int argc, char* argv[])
{
	int n = 12;
	char* file=NULL;
	int algorithm = CPU;
	//int seed = time(NULL);

	
	if (argc==1) {
		printf("SPARSESNP, a project to test sparse matrix representations of SNP on GPUs.\n");
		printf("\nFormat: %s [i] [a]\n",argv[0]);
		printf("Where: \n");
		//printf("\n[i] is the input file describing the SNP to be simulated (check format in help file of repository)\n");
		printf("\n[i] is the example index:\n");
		printf("\t1 = sorting network\n");
		printf("\t2 = \n");
		printf("\n[a] is the algorithm index\n");
		printf("\t1 = CPU (not implemented yet)\n");
		printf("\t1 = GPU lineal algebra CUBLAS\n");
		printf("\t2 = GPU sparse representation CUSPARSE\n");
		printf("\t3 = GPU sparse representation\n");
		return 0;
	}
		
	if (argc>1) {
		file = argv[1];	
	}
	
	if (argc>2) {
		algorithm = atoi(argv[2])-1;
	}
	// Read the input file



	// 
		
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
