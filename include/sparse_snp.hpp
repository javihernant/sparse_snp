#ifndef _SPARSE_SNP_
#define _SPARSE_SNP_

// Algorithms
#define CPU      0
#define GPU_CUBLAS 1
#define GPU_CUSPARSE 2
#define GPU_SPARSEREP 3

// Modes
#define NO_DEBUG 0
#define DEBUG    1

typedef unsigned short int  ushort;
typedef unsigned int        uint;
typedef unsigned char       uchar;

class SNP_model
{
public:
    SNP_model(uint n, uint m);
    ~SNP_model();
    /** Add a rule to neuron nid, 
     * regular expression defined by e_n and e_i, and a^c -> a^p.
     * This must be called sorted by neuron */
    void add_rule (uint nid, uchar e_n, uchar e_i, uchar c, uchar p);
    /** Add synapse from neuron i to j. 
     * This must be called after adding all rules */
    void add_synapse (uint i, uint j);
    /** Load the introduced model to the GPU.
     * The status of model computation gets reset
     */
    void load_to_gpu();
    /** Perform a step on the model. 
     * Returns if no more steps can be done.
     */
    bool step();

protected:
    uint n;                   // number of neurons
    uint m;                   // number of rules

    // CPU part
    ushort * conf_vector;     // configuration vector (# neurons)
    short  * trans_matrix;    // transition matrix (# rules * # neurons), requires negative numbers
    ushort * spiking_vector;  // spiking vector (# neurons)
    uint   * rule_index;      // indicates for each neuron, the starting rule index (# neurons+1)

    struct _rule {
        uchar  * En;
        uchar  * Ei;
        uchar  * c;    
        uchar  * p;
    } rules, d_rules;

    // GPU counterpart
    ushort * d_conf_vector;
    uchar  * d_trans_matrix;
    ushort * d_spiking_vector;
    uint   * d_rule_index;      // indicates for each neuron, the starting rule index (# neurons+1)
};











// Functions for the simulator
#endif
