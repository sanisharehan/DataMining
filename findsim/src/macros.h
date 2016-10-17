/*
 * \file macros.h
 *
 *  Created on: Feb 3, 2014
 *      Author: dragos
 */

#ifndef _FINDSIM_MACROS_H_
#define _FINDSIM_MACROS_H_


/** MACROS  **/

#define F_NEQUALS_P(a,b,p) \
	do { if(fabs((a)-(b) ) > p) return 0; } while(0)



/*-------------------------------------------------------------
 * Usefull commands
 *-------------------------------------------------------------*/
#define da_max(a, b) ((a) >= (b) ? (a) : (b))
#define da_min(a, b) ((a) >= (b) ? (b) : (a))
#define da_max3(a, b, c) ((a) >= (b) && (a) >= (c) ? (a) : ((b) >= (a) && (b) >= (c) ? (b) : (c)))
#define DA_SWAP(a, b, tmp) do {(tmp) = (a); (a) = (b); (b) = (tmp);} while(0)
#define DA_INC_DEC(a, b, val) do {(a) += (val); (b) -= (val);} while(0)
#define da_sign(a, b) ((a >= 0 ? b : -b))
#define da_abs(x) ((x) >= 0 ? (x) : -(x))



/*-------------------------------------------------------------
 * CSR conversion macros
 *-------------------------------------------------------------*/
#define CSRMAKE(i, n, a) \
   do { \
     for (i=1; i<n; i++) a[i] += a[i-1]; \
     for (i=n; i>0; i--) a[i] = a[i-1]; \
     a[0] = 0; \
   } while(0)

#define CSRSHIFT(i, n, a) \
   do { \
     for (i=n; i>0; i--) a[i] = a[i-1]; \
     a[0] = 0; \
   } while(0)

/*********************
 * Progress indicator
 *********************/

#define NPCT    10   /* number of steps */
#define da_progress_init(pct, indicator, niterations) \
    do { \
        indicator = ceil(niterations/(float)NPCT); \
        pct = 0; \
    } while(0)

#define da_progress_init_steps(pct, indicator, niterations, nsteps) \
    do { \
        indicator = ceil(niterations/(float)nsteps); \
        pct = 0; \
    } while(0)

#define da_progress_advance(pct) \
    do { \
        if(pct > 0 && pct < 100) \
            printf("%d%%..", pct); \
        fflush(stdout); \
        pct += NPCT; \
    } while(0)

#define da_progress_advance_steps(pct, nsteps) \
    do { \
        if(pct > 0 && pct < 100) \
            printf("%d%%..", pct); \
        fflush(stdout); \
        pct += 100.0/nsteps; \
    } while(0)

#define da_progress_finalize(pct) \
    do { \
        while(pct < 100){ \
            if(pct > 0) \
                printf("%d%%..", pct); \
            pct += NPCT; \
        } \
        if(pct == 100) \
            printf("%d%%", pct); \
        fflush(stdout); \
    } while(0)

#define da_progress_finalize_steps(pct, nsteps) \
    do { \
        while(pct < 100){ \
            if(pct > 0) \
                printf("%d%%..", pct); \
            pct += 100.0/nsteps; \
        } \
        if(pct == 100) \
            printf("%d%%", pct); \
        fflush(stdout); \
    } while(0)

/**
 * Debug macros
 */

#ifndef NDEBUG
    #define ASSERT(expr)                                          \
    if (!(expr)) {                                               \
        printf("***ASSERTION failed on line %d of file %s: " #expr "\n", \
              __LINE__, __FILE__);                               \
        assert(expr);                                                \
    }

    #define ASSERTP(expr,msg)                                          \
    if (!(expr)) {                                               \
        printf("***ASSERTION failed on line %d of file %s: " #expr "\n", \
              __LINE__, __FILE__);                               \
        printf msg ; \
        printf("\n"); \
        assert(expr);                                                \
    }
#else
    #define ASSERT(expr) ;
    #define ASSERTP(expr,msg) ;
#endif



#endif /* MACROS_H_ */
