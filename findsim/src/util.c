/*!
 \file  util.c
 \brief This file contains utilities for the application

 \author David C. Anastasiu
 */

#include "includes.h"


/*************************************************************************/
/*! This function prints an error message and raises a signum signal
 */
/*************************************************************************/
void da_errexit(const char* const f_str,...)
{
    va_list argp;

    va_start(argp, f_str);
    vfprintf(stderr, f_str, argp);
    va_end(argp);

    fprintf(stderr,"\n");
    fflush(stderr);

    raise(SIGTERM);
}



/*************************************************************************/
/*! If format not specifically given (> 0), check if a text (non-binary) text file
 *  containing a csr is in CLUTO or CSR format.
    \param file is the matrix file to be checked.
    \return the CSR format: DA_FMT_CLUTO or DA_FMT_CSR
 */
/*************************************************************************/
char da_getFileFormat(char *file, const char format)
{
	if(format > 0) return format;
	size_t nnz;
	char fmt;
	char *ext, *p;

	ext = strrchr(file, '.');
	if(ext){
		ext++;
		//make lowercase
		for (p=ext ; *p; ++p) *p = tolower(*p);
		if ((fmt = da_getStringID(fmt_options, ext)) > -1)
			return fmt;
	} else if(da_fexists(file)){ // assume some sort of CSR. Can we guess?
		da_getfilestats(file, NULL, &nnz, NULL, NULL);
		return (nnz%2 == 1) ? DA_FMT_CLUTO : DA_FMT_CSR;
	}
	return -1;
}



#define COMPERRPRINT(ar, ac, av, br, bc, bv, rc, fr) \
	printf("%sa[" PRNT_IDXTYPE "," PRNT_IDXTYPE "," PRNT_VALTYPE \
	"] != b[" PRNT_IDXTYPE "," PRNT_IDXTYPE "," PRNT_VALTYPE "]", fr++?", ":"", rc?(ar):(ac), rc?(ac):(ar), av, rc?(br):(bc), rc?(bc):(br), bv);

#define COMPERRPRINT_A(ar, ac, bv, rc, fr) \
    printf("%s!a[" PRNT_IDXTYPE "," PRNT_IDXTYPE ",(" PRNT_VALTYPE ")]", fr++?", ":"", rc?(ar):(ac), rc?(ac):(ar), bv);

#define COMPERRPRINT_B(br, bc, av, rc, fr) \
	printf("%s!b[" PRNT_IDXTYPE "," PRNT_IDXTYPE ",(" PRNT_VALTYPE ")]", fr++?", ":"", rc?(br):(bc), rc?(bc):(br), av);

#define COMPERRPRINT_VAL(ar, ac, av, bv, rc, fr) \
    printf("%s![" PRNT_IDXTYPE "," PRNT_IDXTYPE ",(" PRNT_VALTYPE ", " PRNT_VALTYPE ")]", fr++?", ":"", rc?(ar):(ac), rc?(ac):(ar), av, bv);


/**
 * Compare two csr matrices and print out differences
 * 	\param doc1 first matrix to compare
 * 	\param doc2 second matrix to compare
 * 	\param eps Float max delta for value comparison
 * 	\param compVals Whether values should be compared
 */
void da_csrCompare(da_csr_t* doc1, da_csr_t* doc2, float eps, char compInds, char compVals){
	ssize_t j, k, ndiff = 0;
	idx_t i, l, fr;
	da_csr_t *a = NULL, *b = NULL;
	ptr_t *ptr1, *ptr2;
	idx_t *ind1, *ind2;
	val_t *val1, *val2;
	char rc;

	ASSERT((doc1->rowptr && doc2->rowptr) || (doc1->colptr && doc2->colptr));

	a = da_csr_Copy(doc1);
	b = da_csr_Copy(doc2);
	if(compInds){
        da_csr_SortIndices(a, DA_ROW);
        da_csr_SortIndices(b, DA_ROW);
	}

	if(a->rowptr){
		rc = 1;
		ptr1 = a->rowptr;
		ind1 = a->rowind;
		val1 = a->rowval;
		ptr2 = b->rowptr;
		ind2 = b->rowind;
		val2 = b->rowval;
	} else {
		rc = 0;
		ptr1 = a->rowptr;
		ind1 = a->rowind;
		val1 = a->rowval;
		ptr2 = b->rowptr;
		ind2 = b->rowind;
		val2 = b->rowval;
	}
	if(!val1) compVals = 0;

	if((a->nrows != b->nrows) || (a->ncols != b->ncols) || (ptr1[a->nrows] != ptr2[b->nrows]))
		printf("Matrix stats differ: A[" PRNT_IDXTYPE "," PRNT_IDXTYPE "," PRNT_PTRTYPE
				"] != B[" PRNT_IDXTYPE "," PRNT_IDXTYPE "," PRNT_PTRTYPE "].\n",
				a->nrows, a->ncols, ptr1[a->nrows],
				b->nrows, b->ncols, ptr2[b->nrows]);
	printf("Differences: \n");
	if(compVals && !compInds){
        for(i=0; i < a->nrows && i < b ->nrows; i++){
            fr=0;
            for(j=ptr1[i], k=ptr2[i]; j < ptr1[i+1] && k < ptr2[i+1]; ++j, ++k){
                if(da_abs(val1[j] - val2[k]) > eps){
                    printf("%s[%d %zu %f %f]", fr++?", ":"", i, j-ptr1[i]+1, val1[j], val2[k]);
                    ndiff++;
                }
            }
            for( ; j < ptr1[i+1]; j++ ){
                printf("%s!b[%d %zu %f]", fr++?", ":"", i, j-ptr1[i]+1, val1[j]);
                ndiff++;
            }
            for( ; k < ptr2[i+1]; k++ ){
                printf("%s!a[%d %zu %f]", fr++?", ":"", i, k-ptr2[i]+1, val2[k]);
                ndiff++;
            }
            if(fr) printf("\n");
        }
        for(l=i; i < a->nrows; i++){
            for(fr=0, j=ptr1[i]; j < ptr1[i+1]; j++ ){
                printf("%s!b[%d %zu %f]", fr++?", ":"", i, j-ptr1[i]+1, val1[j]);
                ndiff++;
            }
            if(fr) printf("\n");
        }
        i=l;
        for( ; i < b->nrows; i++){
            for(fr=0, k=ptr2[i]; k < ptr2[i+1]; k++ ){
                printf("%s!a[%d %zu %f]", fr++?", ":"", i, k-ptr2[i]+1, val2[k]);
                ndiff++;
            }
            if(fr) printf("\n");
        }

	} else {

        for(i=0; i < a->nrows && i < b ->nrows; i++){
            fr=0;
            for(j=ptr1[i], k=ptr2[i]; j < ptr1[i+1] && k < ptr2[i+1]; ){
                if(ind1[j] == ind2[k]){
                    if(compVals && da_abs(val1[j] - val2[k]) > eps){
                        COMPERRPRINT(i+1, ind1[j]+1, val1[j], i+1, ind2[k]+1, val2[k], rc, fr);
                        ndiff++;
                    }
                    j++;
                    k++;
                } else if(ind1[j] > ind2[k]){
                    COMPERRPRINT_A(i+1, ind2[k]+1, val2[k], rc, fr);
                    k++;
                    ndiff++;
                } else {
                    COMPERRPRINT_B(i+1, ind1[j]+1, val1[j], rc, fr);
                    j++;
                    ndiff++;
                }
            }
            for( ; j < ptr1[i+1]; j++ ){
                COMPERRPRINT_B(i+1, ind1[j]+1, val1[j], rc, fr);
                ndiff++;
            }
            for( ; k < ptr2[i+1]; k++ ){
                COMPERRPRINT_A(i+1, ind2[k]+1, val2[k], rc, fr);
                ndiff++;
            }
            if(fr) printf("\n");
        }
        for(l=i; i < a->nrows; i++){
            for(fr=0, j=ptr1[i]; j < ptr1[i+1]; j++ ){
                COMPERRPRINT_B(i+1, ind1[j]+1, val1[j], rc, fr);
                ndiff++;
            }
            if(fr) printf("\n");
        }
        i=l;
        for( ; i < b->nrows; i++){
            for(fr=0, k=ptr2[i]; k < ptr2[i+1]; k++ ){
                COMPERRPRINT_A(i+1, ind2[k]+1, val2[k], rc, fr);
                ndiff++;
            }
            if(fr) printf("\n");
        }

	}

	printf("Overall, %zu differences were encountered between A and B.\n", ndiff);

	da_csr_FreeAll(&a, &b, LTERM);
}



/**
 * Verify results given pre-computed stored results
 * \param ngbrs1 Neighbors found for each row in the input matrix
 * \param ngbrs2 True Neighbors found for each row in the input matrix
 * \param nsz Check only first nsz neighbors
 * \param print_errors Print any errors encountered
 */
void verify_knng_results(da_csr_t *ngbrs1, da_csr_t *ngbrs2, idx_t nsz, char print_errors)
{
    float v, lv, lv2;
    double recall, crecall;
    size_t i, j, k, cid, ln, sz, c, cc, n, nrows, nrows2, ncols, nnz, lnlen, err;
    idx_t progressInd, pct;
    char *line=NULL, *head, *tail;
    FILE *fpin;

    idx_t *ind = ngbrs1->rowind, *ind2 = ngbrs2->rowind;
    val_t *val = ngbrs1->rowval, *val2 = ngbrs2->rowval;
    ptr_t *ptr = ngbrs1->rowptr, *ptr2 = ngbrs2->rowptr;
    nrows = ngbrs1->nrows;

    if(nrows != ngbrs2->nrows)
        da_errexit("Num rows in result %xd does not match that in the verification file %zd.\n",
                nrows, ngbrs2->nrows);

    val_t *row = da_vsmalloc(nrows, -1.0, "row");

    /* compare results, one row at a time */
    recall = crecall = 0.0;
    da_progress_init(pct, progressInd, nrows);
    printf("Checking recall... ");
    for (n=0, i=0; i<nrows; ++i) {
        if(ptr[i+1] == ptr[i]){
            if (i % progressInd == 0 )
                da_progress_advance(pct);
            continue;
        }
        for(lv=FLT_MAX, j=ptr[i]; j < ptr[i+1]; ++j){
            row[ind[j]] = val[j];
            if(val[j] < lv)
                lv = val[j];
        }

        c = cc = err = 0;
        lv2 = FLT_MAX;
        ln = da_min(nsz, ptr2[i+1]-ptr2[i]);
        for(k=0, j=ptr2[i]; j < ptr2[i+1] && k < nsz; ++j, ++k){
            cid = ind2[j];
            v = val2[j];
            if(row[cid] > -1){
                c++;
                if(da_abs(row[cid] - v) < 1e-4){
                    cc++;
                } else if(print_errors > 0){  /* show neighbors we found who's sim may be incorrectly computed */
                    printf("[%zu %zu %f %f] ", i+1, cid+1, v, row[cid]);
                    err++;
                }
                row[cid] = 1;
            } else if(da_abs(lv - v) < 1e-4){
                cc++;
                if(print_errors > 1){ /* show neighbors we did not find within the min values */
                    printf("[%zu *%zu %f] ", i+1, cid+1, v);
                    err++;
                }
            } else if(print_errors > 1){ /* show neighbors we did not find */
                printf("[%zu -%zu %f] ", i+1, cid+1, v);
                err++;
            }
            if(v < lv2)
                lv2 = v;
        }
        if(ln > 0){
            recall += (double)c/(double)ln;
            crecall += (double)cc/(double)ln;
            n++;
        }
        for(k-=err, j=ptr[i]; j < ptr[i+1]; ++j){
            if(print_errors > 2 && k < nsz && row[ind[j]] != 1){  /* show extra neighbors we reported that are not in the true neighborhood */
                printf("[%zu +%d %f] ", i+1, ind[j]+1, row[ind[j]]);
                err++;
                k++;
            }
            row[ind[j]] = -1;
        }
        if(print_errors > 0 && err){
            printf("min: %f %f %.5f\n", lv, lv2, da_abs(lv-lv2));
            fflush(stdout);
        }

        if (i % progressInd == 0 )
            da_progress_advance(pct);
    }
    da_progress_finalize(pct);
    da_free((void **)&row, LTERM);

//    printf("\nRecall: %.4f\n", recall/n);
    printf("\nRecall: %.4f\n", crecall/n);
}
