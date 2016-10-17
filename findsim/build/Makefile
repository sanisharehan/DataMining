######
# Makefile for findsim by David C. Anastasiu
#	Required libraries: m
######

# Define libraries and library directories
LIBDIRS := -L/usr/lib -L/usr/local/lib
LIBS := -lm 
# Source include directories
INC += -I/usr/local/include/

# C flags  -fopt-info-vec-all 
CFLAGS += -c -O3 -msse2 -march=native -ffast-math -fstrict-aliasing $(OMPOPTIONS) -DLINUX -D_FILE_OFFSET_BITS=64 -std=c99 -Wall -Wstrict-aliasing -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-label -Wno-unused-variable -Wno-parentheses -Wsequence-point
# Other compile choices
DEBUG := -DNDEBUG # change to nothing to enable internal debug messages.
RM := rm -rf
EXE := findsim 
CC := gcc

###
# Execute make
###

# Add inputs and outputs from these tool invocations to the build variables 
HEADERS := $(shell ls ../src/*.h)
C_SRCS := $(shell cd ../src && ls *.c)
C_OBJS := $(C_SRCS:%.c=%.o)
C_DEPS := $(C_SRCS:%.c=%.d)

# All Targets
all: findsim

# Objects depend on its source and all headers
%.o: ../src/%.c $(HEADERS)
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	$(CC) $(DEBUG) $(INC) $(CFLAGS) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

# Program depends on objects
findsim: $(C_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C Linker'
	$(CC) $(LIBDIRS) $(OMPOPTIONS) -o $(EXE) $(C_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Clean Target
clean:
	$(RM) *.o *.d $(EXE)
	@echo ' '

# These targets do not produce files
.PHONY: all clean

