/*!
\file  da_mkmemory.h
\brief Templates for memory allocation routines

Ported from George Karypis' GKlib library by David C. Anastasiu, with permission, in Aug 2013.

\date   Started 3/29/07
\author George
*/

#ifndef _FINDSIM_MKMEMORY_H_
#define _FINDSIM_MKMEMORY_H_

#include "memory.h"


#define DA_MKALLOC(PRFX, TYPE)\
/*************************************************************************/\
/*! The macro for da_?malloc()-class of routines */\
/**************************************************************************/\
static TYPE* PRFX ## malloc(\
        const size_t n,\
        const char* const msg)\
{\
  return (TYPE *)da_malloc(sizeof(TYPE)*n, msg);\
}\
\
\
/*************************************************************************/\
/*! The macro for da_?malloc_a()-class of routines */\
/**************************************************************************/\
static TYPE* PRFX ## malloc_a(\
        const size_t n,\
        const char* const msg)\
{\
  return (TYPE *)da_malloc_a(sizeof(TYPE)*n, msg);\
}\
\
\
/*************************************************************************/\
/*! The macro for da_?realloc()-class of routines */\
/**************************************************************************/\
static TYPE* PRFX ## realloc(\
        TYPE *ptr,\
        const size_t n,\
        const char* const msg)\
{\
  return (TYPE *)da_realloc((void *)ptr, sizeof(TYPE)*n, msg);\
}\
\
\
/*************************************************************************/\
/*! The macro for da_?realloc_a()-class of routines */\
/**************************************************************************/\
static TYPE* PRFX ## realloc_a(\
        TYPE *ptr,\
        const size_t n,\
        const char* const msg)\
{\
  return (TYPE *)da_realloc_a((void *)ptr, sizeof(TYPE)*n, msg);\
}\
\
\
/*************************************************************************/\
/*! The macro for da_?set()-class of routines */\
/*************************************************************************/\
static TYPE* PRFX ## set(\
        const size_t n,\
        const TYPE val,\
        TYPE* const x)\
{\
  size_t i;\
\
  for (i=0; i<n; i++)\
    x[i] = val;\
\
  return x;\
}\
\
\
/*************************************************************************/\
/*! The macro for da_?set()-class of routines */\
/*************************************************************************/\
static TYPE* PRFX ## setzero(\
        const size_t n,\
        TYPE* const x)\
{\
  memset(x, 0, sizeof(TYPE)*n); \
\
  return x;\
}\
\
\
/*************************************************************************/\
/*! The macro for da_?copy()-class of routines */\
/*************************************************************************/\
static TYPE* PRFX ## copy(\
        const size_t n,\
        TYPE *a,\
        TYPE *b)\
{\
  return (TYPE *)memmove((void *)b, (void *)a, sizeof(TYPE)*n);\
}\
\
\
/*************************************************************************/\
/*! The macro for da_?smalloc()-class of routines */\
/**************************************************************************/\
static TYPE* PRFX ## smalloc(\
        const size_t n,\
        const TYPE ival,\
        const char* msg)\
{\
  TYPE *ptr;\
\
  ptr = (TYPE *)da_malloc(sizeof(TYPE)*n, msg);\
  if (ptr == NULL) \
    return NULL; \
\
  return PRFX ## set(n, ival, ptr); \
}\
\
\
/*************************************************************************/\
/*! The macro for da_?smalloc_a()-class of routines */\
/**************************************************************************/\
static TYPE* PRFX ## smalloc_a(\
        const size_t n,\
        const TYPE ival,\
        const char* msg)\
{\
  TYPE *ptr;\
\
  ptr = (TYPE *)da_malloc_a(sizeof(TYPE)*n, msg);\
  if (ptr == NULL) \
    return NULL; \
\
  return PRFX ## set(n, ival, ptr); \
}\
\
\
/*************************************************************************/\
/*! The macro for da_?nmalloc()-class of routines (null/zero out memory) */\
/**************************************************************************/\
static TYPE* PRFX ## nmalloc(\
        const size_t n,\
        const char* msg)\
{\
  TYPE *ptr;\
\
  ptr = (TYPE *)da_malloc(sizeof(TYPE)*n, msg);\
  if (ptr == NULL) \
    return NULL; \
  memset(ptr, 0, sizeof(TYPE)*n); \
\
  return ptr; \
}\
\
\
/*************************************************************************/\
/*! The macro for da_?nmalloc_a()-class of routines (null/zero out and align memory) */\
/**************************************************************************/\
static TYPE* PRFX ## nmalloc_a(\
        const size_t n,\
        const char* msg)\
{\
  TYPE *ptr;\
\
  ptr = (TYPE *)da_malloc_a(sizeof(TYPE)*n, msg);\
  if (ptr == NULL) \
    return NULL; \
  memset(ptr, 0, sizeof(TYPE)*n); \
\
  return ptr; \
}\
\
\
/*************************************************************************/\
/*! The macro for da_?matalloc()-class of routines */\
/**************************************************************************/\
static TYPE** PRFX ## matalloc(\
        const size_t ndim1,\
        const size_t ndim2,\
        const char* const msg)\
{\
  size_t i, j;\
  TYPE **matrix;\
\
  matrix = (TYPE **)da_malloc(ndim1*sizeof(TYPE *), msg);\
  if (matrix == NULL) \
    return NULL;\
\
  for (i=0; i<ndim1; i++) { \
    matrix[i] = PRFX ## malloc(ndim2, msg);\
    if (matrix[i] == NULL) { \
      for (j=0; j<i; j++) \
        da_free((void **)&matrix[j], LTERM); \
      return NULL; \
    } \
  }\
\
  return matrix;\
}\
\
\
/*************************************************************************/\
/*! The macro for da_?matalloc_a()-class of routines */\
/**************************************************************************/\
static TYPE** PRFX ## matalloc_a(\
        const size_t ndim1,\
        const size_t ndim2,\
        const char* const msg)\
{\
  size_t i, j;\
  TYPE **matrix;\
\
  matrix = (TYPE **)da_malloc(ndim1*sizeof(TYPE *), msg);\
  if (matrix == NULL) \
    return NULL;\
\
  for (i=0; i<ndim1; i++) { \
    matrix[i] = PRFX ## malloc_a(ndim2, msg);\
    if (matrix[i] == NULL) { \
      for (j=0; j<i; j++) \
        da_free((void **)&matrix[j], LTERM); \
      return NULL; \
    } \
  }\
\
  return matrix;\
}\
\
\
/*************************************************************************/\
/*! The macro for da_?matalloc()-class of routines */\
/**************************************************************************/\
static TYPE** PRFX ## smatalloc(\
        const size_t ndim1,\
        const size_t ndim2,\
        const TYPE value,\
        const char* const msg)\
{\
  size_t i, j;\
  TYPE **matrix;\
\
  matrix = (TYPE **)da_malloc(ndim1*sizeof(TYPE *), msg);\
  if (matrix == NULL) \
    return NULL;\
\
  for (i=0; i<ndim1; i++) { \
    matrix[i] = PRFX ## smalloc(ndim2, value, msg);\
    if (matrix[i] == NULL) { \
      for (j=0; j<i; j++) \
        da_free((void **)&matrix[j], LTERM); \
      return NULL; \
    } \
  }\
\
  return matrix;\
}\
\
\
/*************************************************************************/\
/*! The macro for da_?matalloc_a()-class of routines */\
/**************************************************************************/\
static TYPE** PRFX ## smatalloc_a(\
        const size_t ndim1,\
        const size_t ndim2,\
        const TYPE value,\
        const char* const msg)\
{\
  size_t i, j;\
  TYPE **matrix;\
\
  matrix = (TYPE **)da_malloc(ndim1*sizeof(TYPE *), msg);\
  if (matrix == NULL) \
    return NULL;\
\
  for (i=0; i<ndim1; i++) { \
    matrix[i] = PRFX ## smalloc_a(ndim2, value, msg);\
    if (matrix[i] == NULL) { \
      for (j=0; j<i; j++) \
        da_free((void **)&matrix[j], LTERM); \
      return NULL; \
    } \
  }\
\
  return matrix;\
}\
\
\
/*************************************************************************/\
/*! The macro for da_?nmatalloc()-class of routines (null/zero out memory) */\
/**************************************************************************/\
static TYPE** PRFX ## nmatalloc(\
        const size_t ndim1,\
        const size_t ndim2,\
        const char* const msg)\
{\
  size_t i, j;\
  TYPE **matrix;\
\
  matrix = (TYPE **)da_malloc(ndim1*sizeof(TYPE *), msg);\
  if (matrix == NULL) \
    return NULL;\
\
  for (i=0; i<ndim1; i++) { \
    matrix[i] = PRFX ## nmalloc(ndim2, msg);\
    if (matrix[i] == NULL) { \
      for (j=0; j<i; j++) \
        da_free((void **)&matrix[j], LTERM); \
      return NULL; \
    } \
  }\
\
  return matrix;\
}\
\
\
/*************************************************************************/\
/*! The macro for da_?matalloc_a()-class of routines (null/zero out and align memory) */\
/**************************************************************************/\
static TYPE** PRFX ## nmatalloc_a(\
        const size_t ndim1,\
        const size_t ndim2,\
        const char* const msg)\
{\
  size_t i, j;\
  TYPE **matrix;\
\
  matrix = (TYPE **)da_malloc(ndim1*sizeof(TYPE *), msg);\
  if (matrix == NULL) \
    return NULL;\
\
  for (i=0; i<ndim1; i++) { \
    matrix[i] = PRFX ## nmalloc_a(ndim2, msg);\
    if (matrix[i] == NULL) { \
      for (j=0; j<i; j++) \
        da_free((void **)&matrix[j], LTERM); \
      return NULL; \
    } \
  }\
\
  return matrix;\
}\
\
\
/*************************************************************************/\
/*! The macro for da_?matfree()-class of routines */\
/**************************************************************************/\
static void PRFX ## matfree(\
        TYPE ***r_matrix,\
        const size_t ndim1,\
        const size_t ndim2)\
{\
  size_t i;\
  TYPE **matrix;\
\
  if (*r_matrix == NULL) \
    return; \
\
  matrix = *r_matrix;\
\
  for (i=0; i<ndim1; i++) \
    da_free((void **)&(matrix[i]), LTERM);\
\
  da_free((void **)r_matrix, LTERM);\
}\
\
\
/*************************************************************************/\
/*! The macro for da_?matset()-class of routines */\
/**************************************************************************/\
static void PRFX ## matset(\
        TYPE **matrix,\
        const size_t ndim1,\
        const size_t ndim2,\
        const TYPE value)\
{\
  size_t i, j;\
\
  for (i=0; i<ndim1; i++) {\
    for (j=0; j<ndim2; j++)\
      matrix[i][j] = value;\
  }\
}\


#endif
