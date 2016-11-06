/*!
 \file  proto.h
 \brief This file contains function prototypes

 \author David C. Anastasiu
 */
#ifndef _FINDSIM_PROTO_H_
#define _FINDSIM_PROTO_H_

#include "includes.h"

#ifdef __cplusplus
extern "C"
{
#endif



/* main.c */
void      readInputData(params_t *params);
void      da_testMatricesEqual(params_t *params);
void      da_testRecall(params_t *params);
void      da_matrixInfo(params_t *params);
void      da_matrixIo(params_t *params);
void      freeParams(params_t** params);


/* idxjoin.c */
void      findNeighbors(params_t *params);
void      fast_findNeighbors(params_t *params);
void      dynamic_findNeighbors(params_t *params);

/* util.c */
void      da_errexit(const char* const f_str,...);
char      da_getFileFormat(char *file, char const format);
void      da_csrCompare(da_csr_t *a, da_csr_t *b, float eps, char compInds, char compVals);
void      verify_knng_results(da_csr_t *ngbrs1, da_csr_t *ngbrs2, idx_t nsz, char print_errors);


/* cmdline.c */
void cmdline_parse(params_t *ctrl, int argc, char *argv[]);


/***** Library functions ******/

/* da_string.c */
char*     da_strchr_replace(char* const str, const char* const fromlist, const char* const tolist);
char*     da_strtprune(char* const str, const char* const rmlist);
char*     da_strhprune(char* const str, const char* const rmlist);
char*     da_strtoupper(char* const str);
char*     da_strtolower(char* const str);
char*     da_strdup(const char* const orgstr);
int       da_strcasecmp(const char* const s1, const char* const s2);
int       da_strrcmp(const char* const s1, const char* const s2);
char*     da_time2str(time_t time);
char*     da_getStringKey(const da_StringMap_t* const strmap, char const id);
int       da_getStringID(const da_StringMap_t* const strmap, const char* const key);

/* memory.c */
#include "memory.h"

/* da_io.c */
int       da_fexists(const char* const fname);
int       da_dexists(const char* const fname);
ssize_t   da_getfsize(const char* const fname);
void      da_getfilestats(const char* const fname, size_t* r_nlines, size_t* r_ntokens,
            size_t* r_max_nlntokens, size_t* r_nbytes);
char*     da_getbasename(const char* const path);
char*     da_getextname(const char* const path);
char*     da_getfilename(const char* const path);
char*     da_getpathname(const char* const path);
int       da_mkpath(const char* const path);
int       da_rmpath(const char* const path);
FILE*     da_fopen(const char* const fname, const char* const mode, const char* const msg);
void      da_fclose(FILE* stream);
ssize_t   da_read(int const fd, void *vbuf, const size_t count);
ssize_t   da_getline(char** lineptr, size_t* n, FILE* stream);
char**    da_readfile(const char* const fname, size_t* r_nlines);


/* csr.c  */
da_csr_t*  da_csr_Create();
da_csr_t*  da_csr_Alloc(da_csr_t* const mat, idx_t const nrows, idx_t const ncols, ptr_t const nnz,
              uint const what, bool const values);
void       da_csr_Init(da_csr_t* const mat);
void       da_csr_Free(da_csr_t** const mat);
void       da_csr_FreeAll(da_csr_t** ptr1, ...);
void       da_csr_FreeBase(da_csr_t* const mat, char const type);
void       da_csr_LoadBases(da_csr_t* const csr);
void       da_csr_FreeContents(da_csr_t* const mat);
da_csr_t*  da_csr_Copy(const da_csr_t* const mat);
void       da_csr_Grow(da_csr_t* const mat, const ptr_t newNnz);
da_csr_t*  da_csr_Read(const char* const filename,
              char const format, char readvals, char numbering);
void       da_csr_Write(const da_csr_t* const mat, const char* const filename,
              char const format, char writevals, char numbering);
void       da_csr_PrintInfo(const da_csr_t* const mat, const char* const name, const char* const suffix);
void       da_csr_Print(const da_csr_t* const mat);
char       da_csr_isClutoOrCsr(const char* const file);
void       da_csr_CompactColumns(da_csr_t* const mat);
void       da_csr_CompactRows(da_csr_t* const mat);
void       da_csr_SortIndices(da_csr_t* const mat, char const what);
char       da_csr_CheckSortedIndex(da_csr_t* const mat, char const what);
void       da_csr_SortValues(da_csr_t * const mat, char const what, idx_t const mrl, char const how);
void       da_csr_CreateIndex(da_csr_t * const mat, char const what);
void       da_csr_Normalize(da_csr_t* const mat, char const what, char const norm);
void       da_csr_Scale(da_csr_t* const mat);
char       da_csr_Compare(const da_csr_t* const a, const da_csr_t* const b, const double p);
void       da_csr_Transpose(da_csr_t * const mat);


/* sort.c */

#define DA_MKSORT_PROTO(PRFX, TYPE) \
void     PRFX ## sorti(size_t n, TYPE *base);\
void     PRFX ## sortd(size_t n, TYPE *base);\

DA_MKSORT_PROTO(da_p, ptr_t)
DA_MKSORT_PROTO(da_i, idx_t)
DA_MKSORT_PROTO(da_v, val_t)
DA_MKSORT_PROTO(da_iikv, da_iikv_t)
DA_MKSORT_PROTO(da_pikv, da_pikv_t)
DA_MKSORT_PROTO(da_ivkv, da_ivkv_t)

idx_t da_ivkvkselectd(size_t n, idx_t topk, da_ivkv_t *cand);
idx_t da_ivkvkselecti(size_t n, idx_t topk, da_ivkv_t *cand);



/**
 * Memory allocation functions
 */
// base types: ptr, idx, and val
DA_MKALLOC(da_p,      ptr_t)
DA_MKALLOC(da_i,      idx_t)
DA_MKALLOC(da_v,      val_t)
DA_MKALLOC(da_c,      char)
DA_MKALLOC(da_u,      uint)
DA_MKALLOC(da_z,      ssize_t)
DA_MKALLOC(da_f,      float)
DA_MKALLOC(da_d,      double)
DA_MKALLOC(da_l,      long)
DA_MKALLOC(da_ul,     unsigned long)
DA_MKALLOC(da_iikv,   da_iikv_t)
DA_MKALLOC(da_pikv,   da_pikv_t)
DA_MKALLOC(da_ivkv,   da_ivkv_t)


#ifdef __cplusplus
}
#endif

#endif 
