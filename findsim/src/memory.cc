/*!
 \file  memory.c
 \brief This file contains functions for allocating and freeing memory

Ported from George Karypis' GKlib library by David C. Anastasiu, with permission, in Aug 2013.

 \author David C. Anastasiu
 */

#include "includes.h"


void da_matalloc(
        void*** r_matrix,
        const size_t elmlen,
        const size_t ndim1,
        const size_t ndim2)
{
    size_t i, j;
    void **matrix;

    *r_matrix = NULL;

    if ((matrix = (void **)da_malloc(ndim1*sizeof(void *), "da_matalloc: matrix")) == NULL)
        return;

    for (i=0; i<ndim1; i++) {
        if ((matrix[i] = (void *)da_nmalloc(ndim2*elmlen, "da_matalloc: matrix[i]")) == NULL) {
            for (j=0; j<i; j++)
                da_free((void **)&matrix[j], LTERM);
            return;
        }
    }

    *r_matrix = matrix;
}


void da_matalloc_a(
        void*** r_matrix,
        const size_t elmlen,
        const size_t ndim1,
        const size_t ndim2)
{
    size_t i, j;
    void **matrix;

    *r_matrix = NULL;

    if ((matrix = (void **)da_malloc(ndim1*sizeof(void *), "da_matalloc: matrix")) == NULL)
        return;

    for (i=0; i<ndim1; i++) {
        if ((matrix[i] = (void *)da_nmalloc_a(ndim2*elmlen, "da_matalloc: matrix[i]")) == NULL) {
            for (j=0; j<i; j++)
                da_free((void **)&matrix[j], LTERM);
            return;
        }
    }

    *r_matrix = matrix;
}


void da_matfree(
        void*** r_matrix,
        const size_t ndim1,
        const size_t ndim2)
{
    size_t i;
    void **matrix;

    if ((matrix = *r_matrix) == NULL)
        return;

    for (i=0; i<ndim1; i++)
        da_free((void **)&matrix[i], LTERM);

    da_free((void **)r_matrix, LTERM);

}


void *da_malloc(
        const size_t nbytes,
        const char* const msg)
{
    void *ptr=NULL;

    if (nbytes > 0){
        ptr = (void *)malloc(nbytes);
        if (ptr == NULL) {
            da_errexit("MEMORY ERROR: ***Memory allocation failed for %s. Requested size: %zu bytes",
                    msg, nbytes);
            return NULL;
        }
    } else {
        ptr = (void *)malloc(1);
        if (ptr == NULL) {
            da_errexit("MEMORY ERROR: ***Memory allocation failed for %s. Requested size: 1 byte",
                    msg);
            return NULL;
        }
    }

    return ptr;
}


void *da_malloc_a(
        const size_t nbytes,
        const char* const msg)
{
    void *ptr=NULL;

    if (nbytes > 0){
#ifdef DA_MEMORY_ALIGNMENT
        if (posix_memalign(&ptr, DA_MEMORY_ALIGNMENT, nbytes) || ptr == NULL) {
            da_errexit("MEMORY ERROR: ***Memory allocation failed for %s. Requested size: %zu bytes",
                    msg, nbytes);
            return NULL;
        }
#else
        ptr = (void *)malloc(nbytes);
        if (ptr == NULL) {
            da_errexit("MEMORY ERROR: ***Memory allocation failed for %s. Requested size: %zu bytes",
                    msg, nbytes);
            return NULL;
        }
#endif
    } else {
        ptr = (void *)malloc(1);
        if (ptr == NULL) {
            da_errexit("MEMORY ERROR: ***Memory allocation failed for %s. Requested size: 1 byte",
                    msg);
            return NULL;
        }
    }

    return ptr;
}


void *da_nmalloc(
        const size_t nbytes,
        const char* const msg)
{
    void *ptr;
    ptr = da_malloc(nbytes, msg);
    memset(ptr, 0, nbytes);
    return ptr;
}


void *da_nmalloc_a(
        const size_t nbytes,
        const char* const msg)
{
    void *ptr;
    ptr = da_malloc_a(nbytes, msg);
    memset(ptr, 0, nbytes);
    return ptr;
}


void* da_realloc(
        void* oldptr,
        const size_t nbytes,
        const char* const msg)
{
    void *ptr=NULL;

    if (nbytes > 0){
        ptr = (void *)realloc(oldptr, nbytes);
        if (ptr == NULL) {
            da_errexit("MEMORY ERROR: ***Memory realloc failed for %s. " "Requested size: %zu bytes",
                    msg, nbytes);
            return NULL;
        }
    } else {
        ptr = (void *)realloc(oldptr, 1);
        if (ptr == NULL) {
            da_errexit("MEMORY ERROR: ***Memory realloc failed for %s. " "Requested size: %zu byte",
                    msg);
            return NULL;
        }
    }

    return ptr;
}


void* da_realloc_a(
        void* oldptr,
        const size_t nbytes,
        const char* const msg)
{
    void *ptr=NULL;

    if (nbytes > 0){
        ptr = (void *)realloc(oldptr, nbytes);
        if (ptr == NULL) {
            da_errexit("MEMORY ERROR: ***Memory realloc failed for %s. " "Requested size: %zu bytes",
                    msg, nbytes);
            return NULL;
        }
#ifdef DA_MEMORY_ALIGNMENT
        /** check alignment. if not correct, move to aligned memory block */
        if(MEM_NOT_ALIGNED(ptr, DA_MEMORY_ALIGNMENT)){
            char *msg2 = da_strcat(msg, " :: alignment realloc.");
            void *ptr2 = ptr;
            ptr = da_malloc_a(nbytes, msg2);
            memmove(ptr, ptr2, nbytes);
            da_free((void**)&ptr2, &msg2, LTERM);
        }
#endif
    } else {
        ptr = (void *)realloc(oldptr, 1);
        if (ptr == NULL) {
            da_errexit("MEMORY ERROR: ***Memory realloc failed for %s. " "Requested size: %zu byte",
                    msg);
            return NULL;
        }
    }

    return ptr;
}


void *da_nrealloc(
        void* oldptr,
        const size_t nbytes,
        const char* const msg)
{
    void *ptr;
    ptr = da_realloc(oldptr, nbytes, msg);
    memset(ptr, 0, nbytes);
    return ptr;
}


void *da_nrealloc_a(
        void* oldptr,
        const size_t nbytes,
        const char* const msg)
{
    void *ptr;
    ptr = da_realloc_a(oldptr, nbytes, msg);
    memset(ptr, 0, nbytes);
    return ptr;
}


void da_free(
        void** ptr1,
        ...)
{
    va_list plist;
    void **ptr;

    if (*ptr1 != NULL) {
        free(*ptr1);
    }
    *ptr1 = NULL;

    va_start(plist, ptr1);
    while ((ptr = va_arg(plist, void **)) != LTERM) {
        if (*ptr != NULL) {
            free(*ptr);
        }
        *ptr = NULL;
    }
    va_end(plist);
}

