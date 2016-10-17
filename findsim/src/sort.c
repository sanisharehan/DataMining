/*!
\file  sort.c
\brief This file contains various sorting routines

These routines are implemented using the DASORT macro that is defined
in da_qsort.h and is based on GNU's GLIBC qsort() implementation.

Ported from George Karypis' GKlib library by David C. Anastasiu, with permission, in Aug 2013.

Additional sorting routines can be created using the same way that
these routines where defined.

\date   Started 2/27/2013
\author David C. Anastasiu
 */

#include "includes.h"



/*************************************************************************/
/*! Sorts an array of ptr_t in increasing order */
/*************************************************************************/
void da_psorti(const size_t n, ptr_t* const base)
{
#define ptr_lt(a, b) ((*a) < (*b))
    DA_MKQSORT(ptr_t, base, n, ptr_lt);
#undef ptr_lt
}


/*************************************************************************/
/*! Sorts an array of ptr_t in decreasing order */
/*************************************************************************/
void da_psortd(const size_t n, ptr_t* const base)
{
#define ptr_gt(a, b) ((*a) > (*b))
    DA_MKQSORT(ptr_t, base, n, ptr_gt);
#undef ptr_gt
}


/*************************************************************************/
/*! Sorts an array of idx_t in increasing order */
/*************************************************************************/
void da_isorti(const size_t n, idx_t* const base)
{
#define idx_lt(a, b) ((*a) < (*b))
    DA_MKQSORT(idx_t, base, n, idx_lt);
#undef idx_lt
}


/*************************************************************************/
/*! Sorts an array of idx_t in decreasing order */
/*************************************************************************/
void da_isortd(const size_t n, idx_t* const base)
{
#define idx_gt(a, b) ((*a) > (*b))
    DA_MKQSORT(idx_t, base, n, idx_gt);
#undef idx_gt
}


/*************************************************************************/
/*! Sorts an array of val_t in increasing order */
/*************************************************************************/
void da_vsorti(const size_t n, val_t* const base)
{
#define val_lt(a, b) ((*a) < (*b))
    DA_MKQSORT(val_t, base, n, val_lt);
#undef val_lt
}


/*************************************************************************/
/*! Sorts an array of val_t in decreasing order */
/*************************************************************************/
void da_vsortd(const size_t n, val_t* const base)
{
#define float_gt(a, b) ((*a) > (*b))
    DA_MKQSORT(val_t, base, n, float_gt);
#undef float_gt
}

/*************************************************************************/
/*! Sorts an array of da_pikv_t in increasing order */
/*************************************************************************/
void da_iikvsorti(const size_t n, da_iikv_t* const base)
{
#define da_iikv_lt(a, b) ((a)->val < (b)->val)
    DA_MKQSORT(da_iikv_t, base, n, da_iikv_lt);
#undef da_iikv_lt
}


/*************************************************************************/
/*! Sorts an array of da_iikv_t in decreasing order */
/*************************************************************************/
void da_iikvsortd(const size_t n, da_iikv_t* const base)
{
#define da_iikv_gt(a, b) ((a)->val > (b)->val)
    DA_MKQSORT(da_iikv_t, base, n, da_iikv_gt);
#undef da_iikv_gt
}

/*************************************************************************/
/*! Sorts an array of da_pikv_t in increasing order */
/*************************************************************************/
void da_pikvsorti(const size_t n, da_pikv_t* const base)
{
#define da_pikv_lt(a, b) ((a)->val < (b)->val)
    DA_MKQSORT(da_pikv_t, base, n, da_pikv_lt);
#undef da_pikv_lt
}


/*************************************************************************/
/*! Sorts an array of da_pikv_t in decreasing order */
/*************************************************************************/
void da_pikvsortd(const size_t n, da_pikv_t* const base)
{
#define da_pikv_gt(a, b) ((a)->val > (b)->val)
    DA_MKQSORT(da_pikv_t, base, n, da_pikv_gt);
#undef da_pikv_gt
}


/*************************************************************************/
/*! Sorts an array of da_ivkv_t in increasing order */
/*************************************************************************/
void da_ivkvsorti(const size_t n, da_ivkv_t* const base)
{
#define da_ivkv_lt(a, b) ((a)->val < (b)->val)
    DA_MKQSORT(da_ivkv_t, base, n, da_ivkv_lt);
#undef da_ivkv_lt
}


/*************************************************************************/
/*! Sorts an array of da_ivkv_t in decreasing order */
/*************************************************************************/
void da_ivkvsortd(const size_t n, da_ivkv_t* const base)
{
#define da_ivkv_gt(a, b) ((a)->val > (b)->val)
    DA_MKQSORT(da_ivkv_t, base, n, da_ivkv_gt);
#undef da_ivkv_gt
}


/* Byte-wise swap two items of size SIZE. */
#define DA_QSSWAP(a, b, stmp) do { stmp = (a); (a) = (b); (b) = stmp; } while (0)

/**** kselect functions for ix type kv arrays  ****/

/******************************************************************************/
/*! This function puts the 'topk' largest values in the beginning of the array */
/*******************************************************************************/
idx_t da_ivkvkselectd(size_t n, idx_t topk, da_ivkv_t *cand)
{
    idx_t i, j, lo, hi, mid;
    da_ivkv_t stmp;
    val_t pivot;

    if (n <= topk)
        return n; /* return if the array has fewer elements than we want */

    for (lo=0, hi=n-1; lo < hi;) {
        mid = lo + ((hi-lo) >> 1);

        /* select the median */
        if (cand[lo].val < cand[mid].val)
            mid = lo;
        if (cand[hi].val > cand[mid].val)
            mid = hi;
        else
            goto jump_over;
        if (cand[lo].val < cand[mid].val)
            mid = lo;

        jump_over:
        DA_QSSWAP(cand[mid], cand[hi], stmp);
        pivot = cand[hi].val;

        /* the partitioning algorithm */
        for (i=lo-1, j=lo; j<hi; j++) {
            if (cand[j].val >= pivot) {
                i++;
                DA_QSSWAP(cand[i], cand[j], stmp);
            }
        }
        i++;
        DA_QSSWAP(cand[i], cand[hi], stmp);


        if (i > topk)
            hi = i-1;
        else if (i < topk)
            lo = i+1;
        else
            break;
    }

    return topk;
}


/******************************************************************************/
/*! This function puts the 'topk' smallest values in the beginning of the array */
/*******************************************************************************/
idx_t da_ivkvkselecti(size_t n, idx_t topk, da_ivkv_t *cand)
{
    idx_t i, j, lo, hi, mid;
    da_ivkv_t stmp;
    val_t pivot;

    if (n <= topk)
        return n; /* return if the array has fewer elements than we want */

    for (lo=0, hi=n-1; lo < hi;) {
        mid = lo + ((hi-lo) >> 1);

        /* select the median */
        if (cand[lo].val > cand[mid].val)
            mid = lo;
        if (cand[hi].val < cand[mid].val)
            mid = hi;
        else
            goto jump_over;
        if (cand[lo].val > cand[mid].val)
            mid = lo;

        jump_over:
        DA_QSSWAP(cand[mid], cand[hi], stmp);
        pivot = cand[hi].val;

        /* the partitioning algorithm */
        for (i=lo-1, j=lo; j<hi; j++) {
            if (cand[j].val <= pivot) {
                i++;
                DA_QSSWAP(cand[i], cand[j], stmp);
            }
        }
        i++;
        DA_QSSWAP(cand[i], cand[hi], stmp);


        if (i > topk)
            hi = i-1;
        else if (i < topk)
            lo = i+1;
        else
            break;
    }

    return topk;
}

