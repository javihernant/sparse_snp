#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>


#include <snp_static.hpp> //"../include/snp_static.hpp" // <snp_static.hpp> 

using namespace std;

/** Allocation */
SNP_static_cpu::SNP_static_cpu(uint n, uint m) : SNP_model_cpu(n,m)
{
    // n is num of rows, m is num of colums. 
    
    // done by subclasses
    this->trans_matrix    = (short*)  malloc(sizeof(short)*n*m);
    memset(this->trans_matrix,0,sizeof(short)*n*m);


}

/** Free mem */
SNP_static_cpu::~SNP_static_cpu()
{
    free(this->trans_matrix);
}

void SNP_static_cpu::include_synapse(uint i, uint j)
{
    for (int r = rule_index[i]; r < rule_index[i+1]; r++) {
        trans_matrix[r*n+i] = -rules.c[r];
        if (j<n) trans_matrix[r*n+j] = rules.p[r];
    }
}

void SNP_static_cpu::calc_spiking_vector() 
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


void SNP_static_cpu::calc_transition()
{
    for (int nid=0; nid<n; nid++){
        for (int r=0; r<m; r++){
            conf_vector[nid] += spiking_vector[r] * trans_matrix[r*n+nid];
        }
        
    }
    
}

