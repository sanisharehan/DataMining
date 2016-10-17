/*!
 \file  defs.h
 \brief This file contains various constant and parameter definitions

 \author David C. Anastasiu
 */
#ifndef _FINDSIM_DEFS_H_
#define _FINDSIM_DEFS_H_

/* Versions */
#define PROGRAM_NAME        "findsim"
#define VER_MAJOR           0
#define VER_MINOR           0
#define VER_SUBMINOR        1
#define VER_COMMENT         "initial version"

/** General parameter definitions **/




/** base types **/
typedef int32_t idx_t;
#define IDX_MAX INT32_MAX
#define IDX_MIN INT32_MIN
#define PRNT_IDXTYPE "%d"
#define MAXIDX  (1<<8*sizeof(idx_t)-2)
typedef ssize_t ptr_t;
#define PTR_MAX INT64_MAX
#define PTR_MIN INT64_MIN
#define PRNT_PTRTYPE "%zu"
typedef float val_t;
#define VAL_MAX FLT_MAX
#define VAL_MIN FLT_MIN
#define PRNT_VALTYPE "%f"
typedef unsigned int uint;


/* Command-line option codes */
#define CMD_MODE                10
#define CMD_K                   22
#define CMD_EPSILON             23
#define CMD_FMT_WRITE           32
#define CMD_FMT_WRITE_NUM       33
#define CMD_WRITE_VALS          34
#define CMD_WRITE_NUMBERING     35
#define CMD_FMT_READ            36
#define CMD_FMT_READ_NUM        37
#define CMD_READ_VALS           38
#define CMD_READ_NUMBERING      39
#define CMD_VERIFY              40
#define CMD_VERBOSITY           105
#define CMD_VERSION             109
#define CMD_HELP                110


/* signal end of list of pointers */
#ifndef LTERM
    #define LTERM  (void **) 0
#endif

/* Execution modes */
#define MODE_TESTEQUAL          99  /* Test whether two matrices contain the same values */
#define MODE_IO                 98  /* Transform a matrix from some format into another */
#define MODE_INFO               97  /* Find information about a matrix */
#define MODE_RECALL             96  /* Compute recall given true solution */
#define MODE_IDXJOIN            1   /* IdxJoin */


/* CSR structure components */
#define DA_ROW                  1   /* row-based structure */
#define DA_COL                  2   /* col-based structure */
#define DA_ROWCOL               3   /* both row and col-based */

/* sorting types */
#define DA_SORT_I         1    /* sort in increasing order */
#define DA_SORT_D         2    /* sort in decreasing order */

/* CSR input formats */
#define DA_FMT_CSR          2
#define DA_FMT_METIS        3
#define DA_FMT_CLUTO        1
#define DA_FMT_IJV          6

#endif
