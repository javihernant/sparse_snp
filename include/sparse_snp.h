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

typedef unsigned short int ushort;
typedef unsigned int uint;
typedef unsigned char uchar;

// typedef struct
// {
// // params of the simulation (CPU, GPU, ...)
	
// } SNP_PARAMS;


typedef struct
{
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
    uint   * drule_index;      // indicates for each neuron, the starting rule index (# neurons+1)
    
} SNP_STRUCT;











// Functions for the simulator
#endif
