#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
//#define assert

#include <sparse_snp.hpp>

#include <cuda.h>

using namespace std;

/** Allocation */
SNP_model::SNP_model(uint n, uint m)
{
    this->n = n;
    this->m = m;
    this->conf_vector     = (ushort*) malloc(sizeof(ushort)*n);
    this->spiking_vector  = (ushort*) malloc(sizeof(ushort)*m);
    this->rule_index      = (uint*)   malloc(sizeof(uint)*(n+1));
    this->rules.Ei        = (uchar*)  malloc(sizeof(uchar)*m);
    this->rules.En        = (uchar*)  malloc(sizeof(uchar)*m);
    this->rules.c         = (uchar*)  malloc(sizeof(uchar)*m);
    this->rules.p         = (uchar*)  malloc(sizeof(uchar)*m);

    cudaMalloc(&this->d_conf_vector,   sizeof(ushort)*n);
    cudaMalloc(&this->d_spiking_vector,sizeof(ushort)*m);
    cudaMalloc(&this->d_rule_index,    sizeof(uint)*(n+1));
    cudaMalloc(&this->d_rules.Ei,      sizeof(uchar)*m);
    cudaMalloc(&this->d_rules.En,      sizeof(uchar)*m);
    cudaMalloc(&this->d_rules.c,       sizeof(uchar)*m);
    cudaMalloc(&this->d_rules.p,       sizeof(uchar)*m);

    for (int i = 0; i < m; i++) // for each rule
        this->rule_index[i] = 0;

    this->d_trans_matrix=NULL;
    this->trans_matrix=NULL;
    // done by subclasses
    /*this->trans_matrix    = (short*)  malloc(sizeof(short)*n*m);
    cudaMalloc(&this->d_trans_matrix,  sizeof(short)*n*m);
    for (int i = 0; i < m; i++) // for each row = rule
        for (int j = 0; j<n; j++) // for each column = neuron
            this->trans_matrix[i*n+j] = 0;*/
}

/** Free mem */
SNP_model::~SNP_model()
{
    free(this->conf_vector);
    free(this->spiking_vector);
    if (this->trans_matrix) free(this->trans_matrix);
    free(this->rule_index);
    free(this->rules.Ei);
    free(this->rules.En);
    free(this->rules.c);
    free(this->rules.p);

    cudaFree(this->d_conf_vector);
    cudaFree(this->d_spiking_vector);
    if (this->d_trans_matrix) cudaFree(this->d_trans_matrix);
    cudaFree(this->d_rule_index);
    cudaFree(this->d_rules.Ei);
    cudaFree(this->d_rules.En);
    cudaFree(this->d_rules.c);
    cudaFree(this->d_rules.p);
}

/** Add a rule to neuron nid, regular expression defined by e_n and e_i, and a^c -> a^p.
    Must be called sorted by neuron */
void SNP_model::add_rule (uint nid, uchar e_n, uchar e_i, uchar c, uchar p) 
{
    assert(nid < n);

    if (rule_index[nid+1] == 0) // first rule in neuron
        rule_index[nid+1] = rule_index[nid] + 1; 
    else   // keep accumulation
        rule_index[nid+1] = rule_index[nid+1] - rule_index[nid] + 1; 

    uint rid = rule_index[nid+1]-1;

    rules.Ei[rid] = e_i;
    rules.En[rid] = e_n;
    rules.c[rid]  = c;
    rules.p[rid]  = p;
}

/** Add synapse from neuron i to j. 
    Must be called after adding all rules */
void SNP_model::add_synapse (uint i, uint j) 
{
    // ensure parameters within limits
    assert(i < n && j < n);

    // ensure all rules have been introduced already
    assert(rule_index[n+1]==m);

    // SNP does not allow self-synapses
    assert(i!=j);

    for (int r = rule_index[i]; r < rule_index[i+1]; r++) {
        trans_matrix[r*n+i] = rules.c[r];
        trans_matrix[r*n+j] = rules.p[r];
    }
}

void SNP_model::load_to_gpu () 
{
    //handled by sublcasses
    //cudaMemcpy(d_trans_matrix,  trans_matrix,   sizeof(short)*n*m,  cudaMemcpyHostToDevice);

    cudaMemcpy(d_conf_vector,   conf_vector,    sizeof(ushort)*n,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_spiking_vector,spiking_vector, sizeof(ushort)*m,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_rule_index,    rule_index,     sizeof(uint)*(n+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.Ei,      rules.Ei,       sizeof(uchar)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.En,      rules.En,       sizeof(uchar)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.c,       rules.c,        sizeof(uchar)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.p,       rules.p,        sizeof(uchar)*m,    cudaMemcpyHostToDevice);
}


