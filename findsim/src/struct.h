/*!
 \file  struct.h
 \brief This file contains structures needed in the program

 \author David C. Anastasiu
 */
#ifndef _FINDSIM_STRUCT_H_
#define _FINDSIM_STRUCT_H_

#include "includes.h"

/* define key-value structures */
typedef struct {
  idx_t key;
  idx_t val;
} da_iikv_t;
typedef struct {
  ptr_t key;
  idx_t val;
} da_pikv_t;
typedef struct {
  idx_t key;
  val_t val;
} da_ivkv_t;


// Options
/*-------------------------------------------------------------
 * The following data structure implements a string-2-int mapping
 * table used for parsing command-line options
 *-------------------------------------------------------------*/
typedef struct da_StringMap_t {
    char *name;
    int id;
} da_StringMap_t;

extern const da_StringMap_t mode_options[];
extern const da_StringMap_t fmt_options[];



/*-------------------------------------------------------------
 * The following data structure stores stores a string as a
 * pair of its allocated buffer and the buffer itself.
 *-------------------------------------------------------------*/
typedef struct da_str_t {
    size_t len;
    char *buf;
} da_str_t;


/*-------------------------------------------------------------
 * The following data structure stores a sparse CSR matrix
 *-------------------------------------------------------------*/
typedef struct da_csr_t {
	idx_t nrows, ncols;
	ptr_t *rowptr, *colptr;
	idx_t *rowind, *colind;
	val_t *rowval, *colval;
	val_t *rnorms, *cnorms;
} da_csr_t;


/*************************************************************************/
/*! This data structure stores the various variables that make up the 
 * overall state of the system.                                          */
/*************************************************************************/
typedef struct {
	int32_t verbosity;            /* The reporting verbosity level */
	char mode;                    /* What algorithm to execute */
    int32_t k;                    /* k in K-NN */
	float epsilon;                /* Similarity threshold */

	char stats;                   /* Display additional statistics for the matrix in info mode. */
	float fldelta;                /* Float delta, for testing matrix value equality. */

	char fmtRead;                 /* What format the data is stored as: e.g. DA_FMT_BINROW, DA_FMT_CLUTO, DA_FMT_CSR.*/
	char readVals;
	char readNum;
	char fmtWrite;                /* What format the data should be written in: e.g. DA_FMT_BINROW or DA_FMT_CLUTO.*/
	char writeVals;
	char writeNum;

	char *iFile;                  /* The filestem of the input data CSR matrix file. */
    char *oFile;                  /* The filestem of the output file. */
    char *vFile;                  /* The filestem of the verification file. */
	char *filename;               /* temp space for creating output file names */
    da_csr_t  *docs;              /* Documents structure */
	da_csr_t  *neighbors;         /* Neighbors structure */

	/* internal vars */
	idx_t progressInd;            // progress indicator chunk

	/* timers */
	double timer_global;
	double timer_1;
	double timer_2;
	double timer_3;
	double timer_4;
	double timer_5;
	double timer_6;
	double timer_7;
	double timer_8;
	double timer_9;
} params_t;



#endif 
