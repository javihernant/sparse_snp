#ifndef _SNP_MODEL_STATIC_
#define _SNP_MODEL_STATIC_

#include "snp_model.hpp"

class SNP_static: public SNP_model
{
public:
    SNP_static(uint n, uint m);
    ~SNP_static();
    

protected:
    void printTransMX();
    void include_synapse(uint i, uint j);
    /* Calculates the spiking vector with the current configuration */
    void calc_spiking_vector();
    void load_transition_matrix();
    void calc_transition();

};

class SNP_static_ell: public SNP_model
{
public:
    SNP_static_ell(uint n, uint m);
    ~SNP_static_ell();

protected:
    void printTransMX();
    void include_synapse(uint i, uint j);
    void calc_spiking_vector();
    void init_compressed_matrix();
    void load_transition_matrix();
    void calc_transition();

};

// class SNP_static_optimized: public SNP_model
// {
// public:
//     SNP_static_optimized(uint n, uint m);
//     ~SNP_static_optimized();

// protected:
//     void include_synapse(uint i, uint j);
//     /* Calculates the spiking vector with the current configuration */
//     void calc_spiking_vector();
//     void init_compressed_matrix();
//     void load_transition_matrix();
//     void calc_transition();

// };

// class SNP_static_cpu: public SNP_model_cpu
// {
// public:
//     SNP_static_cpu(uint n, uint m);
    
//     ~SNP_static_cpu();

// protected:
//     void include_synapse(uint i, uint j);
//     /* Calculates the spiking vector with the current configuration */
//     void calc_spiking_vector();
//     void calc_transition();

// };

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
