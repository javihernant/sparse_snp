#ifndef _SNP_MODEL_
#define _SNP_MODEL_

// Modes
#define NO_DEBUG 0
#define DEBUG    1

typedef unsigned short int  ushort;
typedef unsigned int        uint;
typedef unsigned char       uchar;

#include <iostream>
#include <utility>
#include "cublas_v2.h"


class SNP_model
{
public:
    
    SNP_model(uint n, uint m, int mode, bool debug);
    ~SNP_model();

    /** 
     * Set a number of spikes, given by s, in the neuron nid.
     * This method should be used to create the initial configuration.
     * This replaces previous value in that neuron in the configuration */
    void set_spikes (uint nid, uint s);
    /** 
     * Consult number of spikes in neuron nid. */
    uint get_spikes (uint nid);
    /** 
     * Add a rule to neuron nid, 
     * regular expression defined by e_n and e_i, and a^c -> a^p.
     * This must be called sorted by neuron, and before adding synapses */
    void add_rule (uint nid, int e_n, int e_i, int c, int p, uint d);
    /** 
     * Add synapse from neuron i to j. 
     * This must be called after adding all rules */
    void add_synapse (uint i, uint j);
    /** 
     * Perform a transition step on the model. 
     * Returns if no more steps can be done. */
    bool transition_step();
    /** 
     * Simulate a computation of the model. 
     * Optionally, set a limit to l steps */
    void compute(int l=1) { while(l-->=1 && !transition_step()); }; // or
    // void compute(int l=1) { while(l-->=1) transition_step(); };

protected:
    uint n;                   // number of neurons
    uint m;                   // number of rules
    int z;                    //number of rows of trans_matrix_ell
    int ex_mode;              //execution mode. SPARSE, ELL=1, OPTIMIZED=2
    bool debug;
    // CPU part
    int * conf_vector;     // configuration vector (# neurons)
    int  * trans_matrix;   // transition matrix (# rules * # neurons), requires negative numbers
    int * spiking_vector;  // spiking vector
    
    int * delays_vector;  // #rules. Delays for each neuron. 
    int   * rule_index;      // indicates for each neuron, the starting rule index (# neurons+1)
    int * z_vector;

    struct _rule {
        int  * En;          // indicates for each rule, the regular expression multiplicity
        int  * Ei;          // indicates for each rule, the regular expression type
        int  * c;           // indicates for each rule, the LHS
        int  * p;           // indicates for each rule, the RHS
        uint *d;              // indicates for each rule, the delay until it's fired
        uint   * nid;         // indicates for each rule, the corresponding neuron (#rules)
        

    } rules, d_rules;

    // GPU counterpart
    int * d_conf_vector;
    int  * d_trans_matrix;
    int * d_spiking_vector;
    int * d_delays_vector;
    int   * d_rule_index;      // indicates for each neuron, the starting rule index (# neurons+1)



    //////////////////////////////////////CUBLAS variables/////////////////////////////////////////////////
    cublasHandle_t handle;
    
    float * cublas_conf_vector;     // configuration vector (# neurons)
    float  * cublas_trans_matrix;   // transition matrix (# rules * # neurons), requires negative numbers
    float * cublas_spiking_vector;  // spiking vector
    

    // GPU counterpart
    float * d_cublas_conf_vector;
    float * d_cublas_trans_matrix;
    float * d_cublas_spiking_vector;
    float * d_cublas_spiking_vector_aux;   //auxiliar spiking vector for use with cublas 
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    
    // Consistency flags
    bool transMX_printed;
    bool gpu_updated;           // true if GPU copy is updated
    bool cpu_updated;           // true if CPU copy is updated
    bool done_rules;            // true if all rules have been introduced (preventing adding synapses)

    // auxiliary methods
    void printSpikingV();
    void printDelaysV();
    void printConfV();
    /** 
     * Load the introduced model to the GPU.
     * The status of model computation gets reset */
    void load_to_gpu();
    /** 
     * Download information from the GPU. */
    void load_to_cpu();  
    void calc_z();
    

    // auxiliary virtual methods (to be defined in the different simulators)    
    // @override define this method to include a synapse in the transition matrix 
    virtual void include_synapse(uint i, uint j) = 0;
    // @override define this method to send the transition matrix to GPU
    virtual void calc_spiking_vector() = 0;
    virtual void printTransMX() = 0;
    virtual void load_transition_matrix() = 0;
    // @override define this method to compute the transition, once the spiking vector is calculated
    virtual void calc_transition() = 0;


};

//version cpu es una prueba. Se eliminará en un futuro.
// class SNP_model_cpu
// {
// public:
//     SNP_model_cpu(uint n, uint m);
//     ~SNP_model_cpu();

//     /** 
//      * Set a number of spikes, given by s, in the neuron nid.
//      * This method should be used to create the initial configuration.
//      * This replaces previous value in that neuron in the configuration */
//     void set_spikes (uint nid, ushort s);
//     /** 
//      * Consult number of spikes in neuron nid. */
//     ushort get_spikes (uint nid);
//     /** 
//      * Add a rule to neuron nid, 
//      * regular expression defined by e_n and e_i, and a^c -> a^p.
//      * This must be called sorted by neuron, and before adding synapses */
//     void add_rule (uint nid, uchar e_n, uchar e_i, uchar c, uchar p);
//     /** 
//      * Add synapse from neuron i to j. 
//      * This must be called after adding all rules */
//     void add_synapse (uint i, uint j);
//     /** 
//      * Perform a transition step on the model. 
//      * Returns if no more steps can be done. */
//     bool transition_step();
//     /** 
//      * Simulate a computation of the model. 
//      * Optionally, set a limit to l steps */
//     void compute(int l=1) { while(l-->=0 || !transition_step()); };

// //protected:
// public:
//     uint n;                   // number of neurons
//     uint m;                   // number of rules

//     // CPU part
//     ushort * conf_vector;     // configuration vector (# neurons)
//     short  * trans_matrix;    // transition matrix (# rules * # neurons), requires negative numbers
//     ushort * spiking_vector;  // spiking vector (# neurons)
//     uint   * rule_index;      // indicates for each neuron, the starting rule index (# neurons+1)

//     struct _rule {
//         uchar  * En;          // indicates for each rule, the regular expression multiplicity
//         uchar  * Ei;          // indicates for each rule, the regular expression type
//         uchar  * c;           // indicates for each rule, the LHS
//         uchar  * p;           // indicates for each rule, the RHS
//         uint   * nid;         // indicates for each rule, the corresponding neuron (#rules)
//     } rules, d_rules;

//     //Consistency flags
//     bool done_rules;            // true if all rules have been introduced (preventing adding synapses)

//     // auxiliary methods
//     // auxiliary virtual methods (to be defined in the different simulators)    
//     // @override define this method to include a synapse in the transition matrix 
//     virtual void include_synapse(uint i, uint j) = 0;
//     virtual void calc_spiking_vector() = 0;
//     // @override define this method to compute the transition, once the spiking vector is calculated
//     virtual void calc_transition() = 0;
// };



#endif
