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


class SNP_model
{
public:
    SNP_model(uint n, uint m);
    ~SNP_model();

    /** 
     * Set a number of spikes, given by s, in the neuron nid.
     * This method should be used to create the initial configuration.
     * This replaces previous value in that neuron in the configuration */
    void set_spikes (uint nid, ushort s);
    /** 
     * Consult number of spikes in neuron nid. */
    ushort get_spikes (uint nid);
    /** 
     * Add a rule to neuron nid, 
     * regular expression defined by e_n and e_i, and a^c -> a^p.
     * This must be called sorted by neuron, and before adding synapses */
    void add_rule (uint nid, uchar e_n, uchar e_i, uchar c, uchar p);
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
    void compute(int l=1) { while(l-->=0 || transition_step()); };

//protected:
public:
    uint n;                   // number of neurons
    uint m;                   // number of rules
    int z;                    //number of rows of trans_matrix_ell

    // CPU part
    short * conf_vector;     // configuration vector (# neurons)
    short  * trans_matrix;    // transition matrix (# rules * # neurons), requires negative numbers
    std::pair<uint, short>  * trans_matrix_ell;
    ushort * spiking_vector;  // spiking vector (# neurons)
    uint   * rule_index;      // indicates for each neuron, the starting rule index (# neurons+1)

    struct _rule {
        short  * En;          // indicates for each rule, the regular expression multiplicity
        short  * Ei;          // indicates for each rule, the regular expression type
        short  * c;           // indicates for each rule, the LHS
        short  * p;           // indicates for each rule, the RHS
        uint   * nid;         // indicates for each rule, the corresponding neuron (#rules)
    } rules, d_rules;

    // GPU counterpart
    short * d_conf_vector;
    short  * d_trans_matrix;
    std::pair<uint, short>  * d_trans_matrix_ell;
    ushort * d_spiking_vector;
    uint   * d_rule_index;      // indicates for each neuron, the starting rule index (# neurons+1)

    // Consistency flags
    bool gpu_updated;           // true if GPU copy is updated
    bool cpu_updated;           // true if CPU copy is updated
    bool done_rules;            // true if all rules have been introduced (preventing adding synapses)

    // auxiliary methods
    /** 
     * Load the introduced model to the GPU.
     * The status of model computation gets reset */
    void load_to_gpu();
    /** 
     * Download information from the GPU. */
    void load_to_cpu();  
    /**
     * Calculates the spiking vector with the current configuration */
    void calc_spiking_vector();

    // auxiliary virtual methods (to be defined in the different simulators)    
    // @override define this method to include a synapse in the transition matrix 
    virtual void include_synapse(uint i, uint j) = 0;
    // @override define this method to send the transition matrix to GPU
    virtual void load_transition_matrix() = 0;
    // @override define this method to compute the transition, once the spiking vector is calculated
    virtual void calc_transition() = 0;
};

class SNP_model_cpu
{
public:
    SNP_model_cpu(uint n, uint m);
    ~SNP_model_cpu();

    /** 
     * Set a number of spikes, given by s, in the neuron nid.
     * This method should be used to create the initial configuration.
     * This replaces previous value in that neuron in the configuration */
    void set_spikes (uint nid, ushort s);
    /** 
     * Consult number of spikes in neuron nid. */
    ushort get_spikes (uint nid);
    /** 
     * Add a rule to neuron nid, 
     * regular expression defined by e_n and e_i, and a^c -> a^p.
     * This must be called sorted by neuron, and before adding synapses */
    void add_rule (uint nid, uchar e_n, uchar e_i, uchar c, uchar p);
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
    void compute(int l=1) { while(l-->=0 || transition_step()); };

//protected:
public:
    uint n;                   // number of neurons
    uint m;                   // number of rules

    // CPU part
    ushort * conf_vector;     // configuration vector (# neurons)
    short  * trans_matrix;    // transition matrix (# rules * # neurons), requires negative numbers
    ushort * spiking_vector;  // spiking vector (# neurons)
    uint   * rule_index;      // indicates for each neuron, the starting rule index (# neurons+1)

    struct _rule {
        uchar  * En;          // indicates for each rule, the regular expression multiplicity
        uchar  * Ei;          // indicates for each rule, the regular expression type
        uchar  * c;           // indicates for each rule, the LHS
        uchar  * p;           // indicates for each rule, the RHS
        uint   * nid;         // indicates for each rule, the corresponding neuron (#rules)
    } rules, d_rules;

    //Consistency flags
    bool done_rules;            // true if all rules have been introduced (preventing adding synapses)

    // auxiliary methods
     
    /**
     * Calculates the spiking vector with the current configuration */
    void calc_spiking_vector();

    // auxiliary virtual methods (to be defined in the different simulators)    
    // @override define this method to include a synapse in the transition matrix 
    virtual void include_synapse(uint i, uint j) = 0;
    // @override define this method to compute the transition, once the spiking vector is calculated
    virtual void calc_transition() = 0;
};



#endif
