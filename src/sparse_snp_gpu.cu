#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
//#define assert

extern "C" { // nvcc compiles in C++
#include <sparse_snp.h>
}

#include <cuda.h>

typedef unsigned int uint;

using namespace std;

/** Allocation */
void init_vars(SNP_STRUCT **snp, uint n, uint m)
{
    *snp = (SNP_STRUCT*) malloc(sizeof(SNP_STRUCT));
    (*snp)->n = n;
    (*snp)->m = m;
    (*snp)->conf_vector     = (ushort*) malloc(sizeof(ushort)*n);
    (*snp)->spiking_vector  = (ushort*) malloc(sizeof(ushort)*m);
    (*snp)->trans_matrix    = (short*)  malloc(sizeof(short)*n*m);
    (*snp)->rule_index      = (uint*)   malloc(sizeof(uint)*(n+1));
    (*snp)->rules.Ei        = (uchar*) malloc(sizeof(uchar)*m);
    (*snp)->rules.En        = (uchar*) malloc(sizeof(uchar)*m);
    (*snp)->rules.c         = (uchar*) malloc(sizeof(uchar)*m);
    (*snp)->rules.p         = (uchar*) malloc(sizeof(uchar)*m);

    (*snp)->d_conf_vector       = (ushort*) cudaMalloc(sizeof(ushort)*n);
    (*snp)->d_spiking_vector    = (ushort*) cudaMalloc(sizeof(ushort)*m);
    (*snp)->d_trans_matrix      = (short*)  cudaMalloc(sizeof(short)*n*m);
    (*snp)->d_rule_index        = (uint*)   cudamalloc(sizeof(uint)*(n+1));
    (*snp)->d_rules.Ei          = (uchar*) malloc(sizeof(uchar)*m);
    (*snp)->d_rules.En          = (uchar*) malloc(sizeof(uchar)*m);
    (*snp)->d_rules.c           = (uchar*) malloc(sizeof(uchar)*m);
    (*snp)->d_rules.p           = (uchar*) malloc(sizeof(uchar)*m);	

    for (int i = 0; i < m; i++) // for each rule
        (*snp)->rule_index[i] = 0;
    for (int i = 0; i < m; i++) // for each row = rule
        for (int j = 0; j<n; j++) // for each column = neuron
            (*snp)->trans_matrix[i*n+j] = 0;
}

/** Free mem */
void free_memory(SNP_STRUCT *snp)
{
    free(snp->conf_vector);
    free(snp->spiking_vector);
    free(snp->trans_matrix);
    free(snp->rule_index);
    free(snp->rules.Ei);
    free(snp->rules.En);
    free(snp->rules.c);
    free(snp->rules.p);

    cudaFree(snp->d_conf_vector);
    cudaFree(snp->d_spiking_vector);
    cudaFree(snp->d_trans_matrix);
    cudaFree(snp->d_rule_index);
    cudaFree(snp->d_rules.Ei);
    cudaFree(snp->d_rules.En);
    cudaFree(snp->d_rules.c);
    cudaFree(snp->d_rules.p);

    free(snp);
}

/** Add a rule to neuron nid, regular expression defined by e_n and e_i, and a^c -> a^p.
    Must be called sorted by neuron */
void add_rule (SNP_STRUCT *snp, uint nid, uchar e_n, uchar e_i, uchar c, uchar p) 
{
    assert(snp && nid < snp->n);

    if (snp->rule_index[nid+1] == 0) // first rule in neuron
        snp->rule_index[nid+1] = snp->rule_index[nid] + 1; 
    else   // keep accumulation
        snp->rule_index[nid+1] = snp->rule_index[nid+1] - snp->rule_index[nid] + 1; 

    uint rid = snp->rule_index[nid+1]-1;

    snp->rule.Ei[rid] = e_i;
    snp->rule.En[rid] = e_n;
    snp->rule.c[rid]  = c;
    snp->rule.p[rid]  = p;
}

/** Add synapse from neuron i to j. 
    Must be called after adding all rules */
void add_synapse (SNP_STRUCT *snp, uint i, uint j) 
{
    // ensure parameters within limits
    assert(snp && i < snp->n && j < snp->n);

    // ensure all rules have been introduced already
    assert(snp->rule_index[snp->n+1]==snp->m);

    // SNP does not allow self-synapses
    assert(i!=j);

    snp->trans_matrix[r*snp->n+i] = snp->rule.c[r];

    for (int r = snp->rule_index[i]; r < snp->rule_index[i+1]; r++)
        snp->trans_matrix[r*snp->n+j] = snp->rule.p[r];
}

void load_to_gpu (SNP_STRUCT *snp) 
{

}


