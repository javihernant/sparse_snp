#ifndef _SNP_MODEL_STATIC_
#define _SNP_MODEL_STATIC_

#include "snp_model.hpp"

class SNP_static: public SNP_model
{
public:
    SNP_static(uint n, uint m);
    ~SNP_static();

protected:
    void include_synapse(uint i, uint j);
    void load_transition_matrix();
    void calc_transition();

};

/* TODO:

class SNP_static_cusparse: public SNP_model
{
public:
    SNP_static_cusparse(uint n, uint m);
    ~SNP_static_cusparse();

protected:
    void include_synapse(uint i, uint j);
    void load_transition_matrix();
    void calc_transition();
};

class SNP_static_sparse: public SNP_model
{
public:
    SNP_static_sparse(uint n, uint m);
    ~SNP_static_sparse();

protected:
    void include_synapse(uint i, uint j);
    void load_transition_matrix();
    void calc_transition();
};*/

#endif
