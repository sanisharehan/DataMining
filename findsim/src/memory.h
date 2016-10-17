/*
 * memory.h
 *
 *  Created on: Jan 8, 2016
 *      Author: David C. Anastasiu
 */

#ifndef _FINDSIM_MEMORY_H_
#define _FINDSIM_MEMORY_H_

/**
 * MACROS
 */
#define MEM_IS_ALIGNED(ptr, nbytes) \
    (((uintptr_t)(const void *)(ptr)) % (nbytes) == 0)
#define MEM_NOT_ALIGNED(ptr, nbytes) \
    (((uintptr_t)(const void *)(ptr)) % (nbytes) != 0)




#ifndef MEMORY_NO_ALIGNMENT
    #ifndef MEMORY_ALIGNMENT
        #define MEMORY_ALIGNMENT 64u
    #endif
#endif


/**
 * @brief This function allocates a two-dimensional matrix.
 * @note Also zeroes out matrix values. For specific types, da_Tmatalloc fills
 *       matrix with a given value, and da_Tnmatalloc zeroes out matrix,
 *       where T is the type.
 *
 * @param[out] r_matrix Pointer to the 2D matrix that should be allocated
 * @param elmlen Number of bytes for each element in the matrix
 * @param ndim1 The first dimension of the matrix
 * @param ndim2 The second dimension of the matrix
 */
void da_matalloc(
        void*** r_matrix,
        const size_t elmlen,
        const size_t ndim1,
        const size_t ndim2);


/**
 * @brief This function allocates a two-dimensional matrix.
 * @note Also zeroes out matrix values. For specific types, da_Tmatalloc fills
 *       matrix with a given value, and da_Tnmatalloc zeroes out matrix,
 *       where T is the type.
 *
 * @param[out] r_matrix Pointer to the 2D matrix that should be allocated
 * @param elmlen Number of bytes for each element in the matrix
 * @param ndim1 The first dimension of the matrix
 * @param ndim2 The second dimension of the matrix
 */
void da_matalloc_a(
        void*** r_matrix,
        const size_t elmlen,
        const size_t ndim1,
        const size_t ndim2);


/**
 * @brief This function frees a two-dimensional matrix.
 *
 * @param[out] r_matrix Pointer to the 2D matrix that should be freed
 * @param ndim1 The first dimension of the matrix
 * @param ndim2 The second dimension of the matrix
 */
void da_matfree(
        void*** r_matrix,
        const size_t ndim1,
        const size_t ndim2);


/**
 * @brief This function is my wrapper around malloc that allocates one byte of memory,
 *        even if 0 bytes are requested. This is to ensure that checks of returned
 *        values do not lead to NULL due to 0 bytes requested.
 *
 * @param nbytes Number of bytes to allocate
 * @param msg Message that should be printed in case of memory allocation error
 */
void *da_malloc(
        const size_t nbytes,
        const char* const msg);


/**
 * @brief This function is my wrapper around malloc that allocates one byte of memory,
 *        even if 0 bytes are requested. This is to ensure that checks of returned
 *        values do not lead to NULL due to 0 bytes requested.
 * @note  If DA_MEMORY_ALIGNMENT is defined, da_malloc allocates aligned memory,
 *        using posix_memalign with alignment DA_MEMORY_ALIGNMENT, which should
 *        be at least the size of the largest type used in the program, and a
 *        2 multiple of that type.
 *
 * @param nbytes Number of bytes to allocate
 * @param msg Message that should be printed in case of memory allocation error
 */
void *da_malloc_a(
        const size_t nbytes,
        const char* const msg);


/**
 * @brief Same as da_malloc but also specifically zeros out memory.
 *
 * @param nbytes Number of bytes to allocate
 * @param msg Message that should be printed in case of memory allocation error
 */
void *da_nmalloc(
        const size_t nbytes,
        const char* const msg);


/**
 * @brief Same as da_malloc_a but also specifically zeros out memory.
 * @note  See note under da_malloc_a
 *
 * @param nbytes Number of bytes to allocate
 * @param msg Message that should be printed in case of memory allocation error
 */
void *da_nmalloc_a(
        const size_t nbytes,
        const char* const msg);


/**
 * @brief This function is my wrapper around realloc
 *
 * @param oldptr Pointer to memory that should be re-allocated
 * @param nbytes Number of bytes for new allocation
 * @param msg String that should be printed in case of memory allocation error
 */
void* da_realloc(
        void* oldptr,
        const size_t nbytes,
        const char* const msg);


/**
 * @brief This function is my wrapper around realloc that allocates aligned memory
 * @note  If DA_MEMORY_ALIGNMENT is defined, da_realloc_a allocates aligned memory,
 *        using posix_memalign with alignment DA_MEMORY_ALIGNMENT, which should
 *        be at least the size of the largest type used in the program, and a
 *        2 multiple of that type. If a call to realloc does not end in aligned
 *        memory, da_realloc_a will allocate new aligned memory and transfer
 *        the contents of the initial pointer to the new aligned memory.
 *
 * @param oldptr Pointer to memory that should be re-allocated
 * @param nbytes Number of bytes for new allocation
 * @param msg String that should be printed in case of memory allocation error
 */
void* da_realloc_a(
        void* oldptr,
        const size_t nbytes,
        const char* const msg);


/**
 * @brief Same as da_realloc_a but also specifically zeros out memory.
 * @brief This function is my wrapper around realloc
 *
 * @param oldptr Pointer to memory that should be re-allocated
 * @param nbytes Number of bytes for new allocation
 * @param msg String that should be printed in case of memory allocation error
 */
void* da_nrealloc(
        void* oldptr,
        const size_t nbytes,
        const char* const msg);


/**
 * @brief Same as da_realloc_a but also specifically zeros out memory.
 * @brief This function is my wrapper around realloc that allocates aligned memory
 * @note  If DA_MEMORY_ALIGNMENT is defined, da_realloc_a allocates aligned memory,
 *        using posix_memalign with alignment DA_MEMORY_ALIGNMENT, which should
 *        be at least the size of the largest type used in the program, and a
 *        2 multiple of that type. If a call to realloc does not end in aligned
 *        memory, da_realloc_a will allocate new aligned memory and transfer
 *        the contents of the initial pointer to the new aligned memory.
 *
 * @param oldptr Pointer to memory that should be re-allocated
 * @param nbytes Number of bytes for new allocation
 * @param msg String that should be printed in case of memory allocation error
 */
void* da_nrealloc_a(
        void* oldptr,
        const size_t nbytes,
        const char* const msg);


/**
 * @brief This function is my wrapper around free, allows multiple pointers
 *
 * @param ptr1 First pointer to memory that should be freed
 * @param ... More pointers to memory that should be freed
 */
void da_free(
        void** ptr1,
        ...);


#endif /* MEMORY_H_ */
