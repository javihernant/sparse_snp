#include <stdio.h>
#include <stdlib.h>
#include <assert.h> //#define assert
#include <cuda.h>

#include <snp_model.hpp>  // "../include/snp_model.hpp" //
// Algorithms
#define CPU      		0
#define GPU_SPARSE		1
#define GPU_ELL 		2
#define GPU_OPTIMIZED	3
#define GPU_CUBLAS 		4
#define GPU_CUSPARSE 	5

using namespace std;

void checkErr(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s", cudaGetErrorString(err));
    }
}

/** Allocation */

SNP_model::SNP_model(uint n, uint m, int mode)
{
    // allocation in CPU
    this->m = m;  // number of rules
    this->n = n;  // number of neurons
    this->ex_mode = mode;

    this->conf_vector     = (int*) malloc(sizeof(int)*n); // configuration vector (only one, we simulate just a computation)
    this->spiking_vector  = NULL; // spiking vector
    this->delays_vector = (int*) malloc(sizeof(int)*(n));
    this->rule_index      = (int*)   malloc(sizeof(int)*(n+1)); // indices of rules inside neuron (start index per neuron)
    this->rules.Ei        = (int*)  malloc(sizeof(int)*m); // Regular expression Ei of a rule
    this->rules.En        = (int*)  malloc(sizeof(int)*m); // Regular expression En of a rule
    this->rules.c         = (int*)  malloc(sizeof(int)*m); // LHS of rule
    this->rules.p         = (int*)  malloc(sizeof(int)*m); // RHS of rule
    this->rules.d = (uint*) malloc(sizeof(uint)*(m));
    this->rules.nid       = (uint*)   malloc(sizeof(uint)*(m)); // Index of the neuron where the rule is

    // initialization (only in CPU, having updated version)
    memset(this->conf_vector,   0,  sizeof(int)*n);
    // memset(this->spiking_vector,0,  sizeof(ushort)*m);
    memset(this->delays_vector,0,  sizeof(int)*n);
    memset(this->rule_index,    -1,  sizeof(int)*(n+1));
    rule_index[0]=0;
    memset(this->rules.Ei,      0,  sizeof(int)*m);
    memset(this->rules.En,      0,  sizeof(int)*m);
    memset(this->rules.c,       0,  sizeof(int)*m);
    memset(this->rules.p,       0,  sizeof(int)*m);
    memset(this->rules.d,     0,  sizeof(uint)*(m));
    memset(this->rules.nid,     0,  sizeof(uint)*(m));

    this->d_trans_matrix=NULL;
    this->trans_matrix=NULL;

    // allocation in GPU
    cudaMalloc(&this->d_conf_vector,   sizeof(int)*n);
    // cudaMalloc(&this->d_spiking_vector,sizeof(ushort)*m);
    cudaMalloc(&this->d_delays_vector,sizeof(int)*n);
    cudaMalloc(&this->d_rule_index,    sizeof(int)*(n+1));
    cudaMalloc(&this->d_rules.Ei,      sizeof(int)*m);
    cudaMalloc(&this->d_rules.En,      sizeof(int)*m);
    cudaMalloc(&this->d_rules.c,       sizeof(int)*m);
    cudaMalloc(&this->d_rules.p,       sizeof(int)*m);
    cudaMalloc(&this->d_rules.d,       sizeof(uint)*m);
    cudaMalloc(&this->d_rules.nid,     sizeof(uint)*m);

    
   
    // memory consistency, who has the updated copy?
    gpu_updated = false; cpu_updated = true;
    done_rules = false;
}

/** Free mem */
SNP_model::~SNP_model()
{
    free(this->conf_vector);
    // free(this->spiking_vector);
    // if (this->trans_matrix) free(this->trans_matrix);
    free(this->delays_vector);
    free(this->rule_index);
    free(this->rules.Ei);
    free(this->rules.En);
    free(this->rules.c);
    free(this->rules.p);
    free(this->rules.d);
    free(this->rules.nid);

    cudaFree(this->d_conf_vector);
    // cudaFree(this->d_spiking_vector);
    // if (this->d_trans_matrix) cudaFree(this->d_trans_matrix);
    cudaFree(this->d_delays_vector);
    cudaFree(this->d_rule_index);
    cudaFree(this->d_rules.Ei);
    cudaFree(this->d_rules.En);
    cudaFree(this->d_rules.c);
    cudaFree(this->d_rules.p);
    cudaFree(this->d_rules.d);
    cudaFree(this->d_rules.nid);
}

void SNP_model::set_spikes (uint nid, uint s)
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (gpu_updated && !cpu_updated) {
        load_to_cpu();
        cpu_updated=true;
    }
    gpu_updated = false;
    //////////////////////////////////////////////////////

    conf_vector[nid] = s;    
}

uint SNP_model::get_spikes (uint nid)
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (gpu_updated && !cpu_updated) {
        load_to_cpu();
        cpu_updated=true;
    }
    //////////////////////////////////////////////////////

    return conf_vector[nid];
}

/** Add a rule to neuron nid, regular expression defined by e_n and e_i, and a^c -> a^p.
    Must be called sorted by neuron */
void SNP_model::add_rule (uint nid, int e_n, int e_i, int c, int p, uint d) 
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    assert(!done_rules);
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    gpu_updated = false; cpu_updated = true;
    //////////////////////////////////////////////////////

    if(rule_index[nid+1]==-1){
        rule_index[nid+1] = rule_index[nid] + 1;
    }else{
        rule_index[nid+1] = rule_index[nid] + (rule_index[nid+1] - rule_index[nid] + 1);
    }
    
    

 
    uint rid = rule_index[nid+1]-1;
    // printf("rule_index[%d]=%d, rule_index[%d]=%d",nid,rule_index[nid], nid+1,rule_index[nid+1]);

    rules.Ei[rid] = e_i;
    rules.En[rid] = e_n;
    rules.c[rid]  = c;
    rules.p[rid]  = p;
    rules.d[rid] = d;
    // if(rid==4){
    //     printf("rules.d[4]=%d",rules.d[4]);
    // }
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
    
    // assert(rule_index[n]==m); //TODO: What if neuron n does not contain rules. Sometimes rule_index ends before index n.
    // SNP does not allow self-synapses
    assert(i!=j);
    done_rules = true; // from now on, no more rules can be added
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    gpu_updated = false; cpu_updated = true;
    //////////////////////////////////////////////////////

    include_synapse(i,j);
}

void SNP_model::calc_z(){
    assert(trans_matrix != NULL);
    z=0;
    for(int i=0; i<m;i++){
        int z_aux =0;
        for(int j=0; j<n;j++){
            if(trans_matrix[i*n+j]>0){
                z_aux++;
            }
        }
        if(z_aux>z){
            z=z_aux;
        }

    }
}

void SNP_model::printAllVecs(){

    int spv_size= ex_mode == GPU_OPTIMIZED ? n : m;
    printf("spiking_vector= ");
    for(int i=0; i<spv_size; i++){
        printf("{%d}",spiking_vector[i]);
    }
    printf("\n");

    printf("delays_vector= "); 
    for(int i=0; i<n; i++){
        printf("{%d}",delays_vector[i]);
    }
    printf("\n");


    printf("conf_vector (after transition)= "); 
    for(int i=0; i< n; i++){
        printf("{%d}",conf_vector[i]);
    }
    printf("\n---------------------------------------\n");
}

bool SNP_model::transition_step()
{
   //returns true if stop criterion reached; false otherwise
    //////////////////////////////////////////////////////
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (!gpu_updated) load_to_gpu();
    cpu_updated = false;
    //////////////////////////////////////////////////////
    
    if(!transMX_printed){
        printTransMX();
        transMX_printed = true;
    }
    calc_spiking_vector(); //after this method is executed, an outdated version of spiking_vec and delay_vec is sent to host

    int spv_size= ex_mode == GPU_OPTIMIZED ? n : m;
    
    

    for(int i=0; i<spv_size; i++){
        // Check if at least one rule is active. If so, continue calculating (return false)

        bool calc_next_trans;
        switch(ex_mode)
        {
        case (GPU_OPTIMIZED):
            calc_next_trans = spiking_vector[i] != -1 || delays_vector[rules.nid[i]]>0;
            break;
        default:
            calc_next_trans = spiking_vector[i] != 0 || delays_vector[rules.nid[i]]>0;
            
        }

        if(calc_next_trans){
            calc_transition();
            load_to_cpu(); 
            printAllVecs();
            
            return false;
        }
    }

    printf("\n\n");


    
    //stop criterion if no rule found active and all neurons are open 
    return true;
}

void SNP_model::load_to_gpu () 
{
    //////////////////////////////////////////////////////
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (gpu_updated) return;
    gpu_updated = true;
    //////////////////////////////////////////////////////

    cudaMemcpy(d_conf_vector,   conf_vector,    sizeof(int)*n,   cudaMemcpyHostToDevice);
    // cudaMemcpy(d_spiking_vector,spiking_vector, sizeof(ushort)*m,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_rule_index,    rule_index,     sizeof(int)*(n+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.Ei,      rules.Ei,       sizeof(int)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.En,      rules.En,       sizeof(int)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.c,       rules.c,        sizeof(int)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.p,       rules.p,        sizeof(int)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.d,       rules.d,        sizeof(uint)*m,    cudaMemcpyHostToDevice);
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
    cudaError_t error;

    
    error = cudaMemcpy(conf_vector, d_conf_vector, sizeof(int)*n, cudaMemcpyDeviceToHost);
    // cudaMemcpy(spiking_vector, d_spiking_vector,  sizeof(ushort)*m, cudaMemcpyDeviceToHost);
    // cudaMemcpy(delays_vector, d_delays_vector,  sizeof(ushort)*n, cudaMemcpyDeviceToHost);
    
    checkErr(error);
}





