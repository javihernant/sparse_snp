#ifndef _SNP_MODEL_STATIC_CPU_
#define _SNP_MODEL_STATIC_CPU_

#include "snp_model_cpu.hpp"

class SNP_static_cpu: public SNP_model_cpu
{
public:
    SNP_static_cpu(uint n, uint m);
    ~SNP_static_cpu();

protected:
    void include_synapse(uint i, uint j);
    void calc_transition();

};

#endif
