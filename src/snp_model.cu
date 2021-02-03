#include <stdio.h>
#include <stdlib.h>
#include <assert.h> //#define assert
#include <cuda.h>

#include <snp_model.hpp>

using namespace std;

/** Allocation */
SNP_model::SNP_model(uint n, uint m)
{
    // allocation in CPU
    this->n = n;  // number of neurons
    this->m = m;  // number of rules
    this->conf_vector     = (ushort*) malloc(sizeof(ushort)*n); // configuration vector (only one, we simulate just a computation)
    this->spiking_vector  = (ushort*) malloc(sizeof(ushort)*m); // spiking vector
    this->rule_index      = (uint*)   malloc(sizeof(uint)*(n+1)); // indeces of rules inside neuron (start index per neuron)
    this->rules.Ei        = (uchar*)  malloc(sizeof(uchar)*m); // Regular expression Ei of a rule
    this->rules.En        = (uchar*)  malloc(sizeof(uchar)*m); // Regular expression En of a rule
    this->rules.c         = (uchar*)  malloc(sizeof(uchar)*m); // LHS of rule
    this->rules.p         = (uchar*)  malloc(sizeof(uchar)*m); // RHS of rule
    this->rules.nid       = (uint*)   malloc(sizeof(uint)*(m)); // Index of the neuron where the rule is

    // allocation in GPU
    cudaMalloc(&this->d_conf_vector,   sizeof(ushort)*n);
    cudaMalloc(&this->d_spiking_vector,sizeof(ushort)*m);
    cudaMalloc(&this->d_rule_index,    sizeof(uint)*(n+1));
    cudaMalloc(&this->d_rules.Ei,      sizeof(uchar)*m);
    cudaMalloc(&this->d_rules.En,      sizeof(uchar)*m);
    cudaMalloc(&this->d_rules.c,       sizeof(uchar)*m);
    cudaMalloc(&this->d_rules.p,       sizeof(uchar)*m);
    cudaMalloc(&this->d_rules.nid,     sizeof(uint)*m);

    // initialization (only in CPU, having updated version)
    memset(this->conf_vector,   0,  sizeof(ushort)*n);
    memset(this->spiking_vector,0,  sizeof(uint)*m);
    memset(this->rule_index,    0,  sizeof(uint)*(n+1));
    memset(this->rules.Ei,      0,  sizeof(uchar)*m);
    memset(this->rules.En,      0,  sizeof(uchar)*m);
    memset(this->rules.c,       0,  sizeof(uchar)*m);
    memset(this->rules.p,       0,  sizeof(uchar)*m);
    memset(this->rules.nid,     0,  sizeof(uint)*(m));

    this->d_trans_matrix=NULL;
    this->trans_matrix=NULL;
   
    // memory consistency, who has the updated copy?
    gpu_updated = false; cpu_updated = true;
    done_rules = false;
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
    free(this->rules.nid);

    cudaFree(this->d_conf_vector);
    cudaFree(this->d_spiking_vector);
    if (this->d_trans_matrix) cudaFree(this->d_trans_matrix);
    cudaFree(this->d_rule_index);
    cudaFree(this->d_rules.Ei);
    cudaFree(this->d_rules.En);
    cudaFree(this->d_rules.c);
    cudaFree(this->d_rules.p);
    cudaFree(this->d_rules.nid);
}

void SNP_model::set_spikes (uint nid, ushort s)
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (gpu_updated && !cpu_updated) load_to_cpu();
    gpu_updated = false;
    //////////////////////////////////////////////////////

    conf_vector[nid] = s;    
}

ushort SNP_model::get_spikes (uint nid)
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (gpu_updated && !cpu_updated) load_to_cpu();
    //////////////////////////////////////////////////////

    return conf_vector[nid];
}

/** Add a rule to neuron nid, regular expression defined by e_n and e_i, and a^c -> a^p.
    Must be called sorted by neuron */
void SNP_model::add_rule (uint nid, uchar e_n, uchar e_i, uchar c, uchar p) 
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    assert(!done_rules);
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    gpu_updated = false; cpu_updated = true;
    //////////////////////////////////////////////////////

    if (rule_index[nid+1] == 0) // first rule in neuron
        rule_index[nid+1] = rule_index[nid] + 1; 
    else   // keep accumulation
        rule_index[nid+1] = rule_index[nid+1] - rule_index[nid] + 1; 

    uint rid = rule_index[nid+1]-1;

    rules.Ei[rid] = e_i;
    rules.En[rid] = e_n;
    rules.c[rid]  = c;
    rules.p[rid]  = p;
    rules.nid[rid]= nid;
}

/** Add synapse from neuron i to j. 
    Must be called after adding all rules */
void SNP_model::add_synapse (uint i, uint j) 
{
    //////////////////////////////////////////////////////
    // ensure parameters within limits
    assert(i < n && j < n);
    // ensure all rules have been introduced already
    assert(rule_index[n+1]==m);
    // SNP does not allow self-synapses
    assert(i!=j);
    done_rules = true; // from now on, no more rules can be added
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    gpu_updated = false; cpu_updated = true;
    //////////////////////////////////////////////////////

    include_synapse(i,j);
}

bool SNP_model::transition_step ()
{
    //////////////////////////////////////////////////////
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (!gpu_updated) load_to_gpu();
    cpu_updated = false;
    //////////////////////////////////////////////////////

    calc_spiking_vector();
    calc_transition();

    return true; // TODO: check if a stopping criterion has been reached
}

void SNP_model::load_to_gpu () 
{
    //////////////////////////////////////////////////////
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (gpu_updated) return;
    gpu_updated = true;
    //////////////////////////////////////////////////////

    cudaMemcpy(d_conf_vector,   conf_vector,    sizeof(ushort)*n,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_spiking_vector,spiking_vector, sizeof(ushort)*m,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_rule_index,    rule_index,     sizeof(uint)*(n+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.Ei,      rules.Ei,       sizeof(uchar)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.En,      rules.En,       sizeof(uchar)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.c,       rules.c,        sizeof(uchar)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.p,       rules.p,        sizeof(uchar)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.nid,     rules.nid,      sizeof(uint)*m,     cudaMemcpyHostToDevice);

    load_transition_matrix();
}

void SNP_model::load_to_cpu ()
{
    //////////////////////////////////////////////////////
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (cpu_updated) return;
    cpu_updated = true;
    //////////////////////////////////////////////////////

    cudaMemcpy(conf_vector,     d_conf_vector,  sizeof(ushort)*n,   cudaMemcpyHostToDevice);
}

__global__ void kalc_spiking_vector(ushort* spiking_vector, ushort* conf_vector, uint* rnid, uchar* rei, uchar* ren, uint m)
{
    uint r = threadIdx.x+blockIdx.x*blockDim.x;
    if (r<m) {
        uchar i = rei[r];
        uchar n = ren[r];
        ushort x = conf_vector[rnid[r]]; // map rule to neuron
        spiking_vector[r] = (ushort) (i&(x==n)) || ((1-i)&(x>=n)); // checking the regular expression
    }
}

void SNP_model::calc_spiking_vector() 
{
    uint bs = 256;
    uint gs = (m+255)/256;
    
    kalc_spiking_vector<<<gs,bs>>>(d_spiking_vector, d_conf_vector, d_rules.nid, d_rules.Ei, d_rules.En, m);
    cudaDeviceSynchronize();
}

