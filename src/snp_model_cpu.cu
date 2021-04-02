#include <stdio.h>
#include <stdlib.h>
#include <assert.h> //#define assert
#include <cuda.h>

#include <snp_model_cpu.hpp> 

using namespace std;

/** Allocation */
SNP_model_cpu::SNP_model_cpu(uint n, uint m)
{
    // allocation in CPU
    this->m = m;  // number of rules
    this->n = n;  // number of neurons
    this->conf_vector     = (ushort*) malloc(sizeof(ushort)*n); // configuration vector (only one, we simulate just a computation)
    this->spiking_vector  = (ushort*) malloc(sizeof(ushort)*m); // spiking vector
    this->rule_index      = (uint*)   malloc(sizeof(uint)*(n+1)); // indeces of rules inside neuron (start index per neuron)
    this->rules.Ei        = (uchar*)  malloc(sizeof(uchar)*m); // Regular expression Ei of a rule
    this->rules.En        = (uchar*)  malloc(sizeof(uchar)*m); // Regular expression En of a rule
    this->rules.c         = (uchar*)  malloc(sizeof(uchar)*m); // LHS of rule
    this->rules.p         = (uchar*)  malloc(sizeof(uchar)*m); // RHS of rule
    this->rules.nid       = (uint*)   malloc(sizeof(uint)*(m)); // Index of the neuron where the rule is

    memset(this->conf_vector,   0,  sizeof(ushort)*n);
    memset(this->spiking_vector,0,  sizeof(uint)*m);
    memset(this->rule_index,    0,  sizeof(uint)*(n+1));
    memset(this->rules.Ei,      0,  sizeof(uchar)*m);
    memset(this->rules.En,      0,  sizeof(uchar)*m);
    memset(this->rules.c,       0,  sizeof(uchar)*m);
    memset(this->rules.p,       0,  sizeof(uchar)*m);
    memset(this->rules.nid,     0,  sizeof(uint)*(m));

    this->trans_matrix=NULL;
    done_rules = false;

}

/** Free mem */
SNP_model_cpu::~SNP_model_cpu()
{
    free(this->conf_vector);
    free(this->spiking_vector);
    // if (this->trans_matrix) free(this->trans_matrix);
    free(this->rule_index);
    free(this->rules.Ei);
    free(this->rules.En);
    free(this->rules.c);
    free(this->rules.p);
    free(this->rules.nid);

}

void SNP_model_cpu::set_spikes (uint nid, ushort s)
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    //////////////////////////////////////////////////////

    conf_vector[nid] = s;    
}

ushort SNP_model_cpu::get_spikes (uint nid)
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    //////////////////////////////////////////////////////

    return conf_vector[nid];
}

/** Add a rule to neuron nid, regular expression defined by e_n and e_i, and a^c -> a^p.
    Must be called sorted by neuron */
void SNP_model_cpu::add_rule (uint nid, uchar e_n, uchar e_i, uchar c, uchar p) 
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    assert(!done_rules);
    //////////////////////////////////////////////////////

    if (rule_index[nid+1] == 0) // first rule in neuron
        rule_index[nid+1] = rule_index[nid] + 1; 
    else   // keep accumulation
        rule_index[nid+1] = rule_index[nid] + rule_index[nid+1] - rule_index[nid] + 1; 

 
    uint rid = rule_index[nid+1]-1;

    rules.Ei[rid] = e_i;
    rules.En[rid] = e_n;
    rules.c[rid]  = c;
    rules.p[rid]  = p;
    rules.nid[rid]= nid;
}

/** Add synapse from neuron i to j. 
    Must be called after adding all rules */
void SNP_model_cpu::add_synapse (uint i, uint j) 
{
    //////////////////////////////////////////////////////
    // ensure parameters within limits
    assert(i < n && j < n+1);
    // ensure all rules have been introduced already
    assert(rule_index[n]==m);
    // SNP does not allow self-synapses
    assert(i!=j);
    done_rules = true; // from now on, no more rules can be added
    //////////////////////////////////////////////////////

    include_synapse(i,j);
}

bool SNP_model_cpu::transition_step ()
{

    calc_spiking_vector();
    calc_transition();

    return true; // TODO: check if a stopping criterion has been reached
}

void SNP_model_cpu::calc_spiking_vector() 
{
    //prueba: activar la primera regla de la neurona que cumpla la expresion regular
    //TODO: escoger una al azar.
    for (int ni=0; ni<n; ni++){
        //vector<int> active_rule_idxs_ni;
        for (int r=rule_index[ni]; r<rule_index[ni+1]; r++){
            uchar i = rules.Ei[r];
            uchar n = rules.En[r];
            ushort x = conf_vector[rules.nid[r]];
            if ((ushort) (i&(x==n)) || ((1-i)&(x>=n))){
                //active_ridx.push_back(r);
                spiking_vector[r] = 1;
                break;
            }
        }
        //get_random(active_rule_idxs_ni);


    }


}

