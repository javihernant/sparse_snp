#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sparse_snp.h>

#include <omp.h>

void init_params()
{
	

}


void init_vars(SNP_STRUCT **snp, int n, int m)
{
    *snp = (SNP_STRUCT*) malloc(sizeof(SNP_STRUCT*));
    (*snp)->n = n;
    (*snp)->m = m;
    (*snp)->conf_vector = malloc(sizeof(ushort)*n);
    (*snp)->spiking_vector = malloc(sizeof(ushort)*m);
    (*snp)->trans_matrix = malloc(sizeof(ushort)*n*m);
}

void free_memory()
{

}

