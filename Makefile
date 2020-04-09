ODIR = obj
SDIR = src
LDIR = lib
BDIR = bin
IDIR = include

CC=g++
RM=rm

OBJ = main.o
OBJ += snp_model.o snp_static.o
BIN = ssnp
#OBJ_LIB = snp_model.o
#LIB = 

CFlags=-c -Wall -fopenmp 
LDFlags= -lm -fopenmp

############ NVIDIA specifics
NCC=nvcc -ccbin=$(CC)
#GENCODE_SM20    := -gencode arch=compute_20,code=\"sm_20,compute_20\"
GENCODE_SM60    := -gencode arch=compute_60,code=\"sm_60,compute_60\"
#GENCODE_SM61    := -gencode arch=compute_61,code=\"sm_61,compute_61\"
GENCODE_SM75    := -gencode arch=compute_75,code=\"sm_75,compute_75\"
GENCODE_FLAGS   := $(GENCODE_SM20) $(GENCODE_SM35) $(GENCODE_SM60)\
                   $(GENCODE_SM61) $(GENCODE_SM75)
NCFlags=-c --compiler-options -Wall -Xcompiler -fopenmp $(GENCODE_FLAGS)
NLDFlags= -lm -Xcompiler -fopenmp 
############

############ Options for GPU and debugging
XCC=$(NCC) 	
XLD=$(NLDFlags)

ifeq ($(dbg),1)
	CFlags += -O0 -g
	NCFlags += -O0 -g
else	
	CFlags += -O3
	NCFlags += -O3
endif
############

all: $(OBJ) $(BIN) $(LIB)

$(LIB): $(patsubst %,$(ODIR)/%,$(OBJ_LIB))
	@mkdir -p $(LDIR)
	ar rcs $(LDIR)/$@ $^ 

$(BIN): $(patsubst %,$(ODIR)/%,$(OBJ))
	@mkdir -p $(BDIR)
	$(XCC) $^ $(XLD) -o $(BDIR)/$@ 

%.o: $(SDIR)/%.cpp
	@mkdir -p $(ODIR)
	$(CC) $(CFlags) -I$(IDIR) -o $(ODIR)/$@ $<

%.o: $(SDIR)/%.cu
	@mkdir -p $(ODIR)
	$(NCC) $(NCFlags) -I$(IDIR) -o $(ODIR)/$@ $<

clean:
	$(RM) $(patsubst %,$(ODIR)/%,$(OBJ)) $(BDIR)/$(BIN) $(LDIR)/$(LIB)
