/*!
 \file  fasterjoin.c

 This file is an attempt to improve performance over the existing index join.

 \brief This file contains IdxJoin related functions. IdxJoin is equivalent to a smart linear search,
 where each query document is compared only against other documents (candidates) that have at least
 one feature in common with the query. All query-candidate similarities are fully computed before
 sorting the results and retaining only results that should be part of the output (top-$k$ results
 with at least $\epsilon$ similarity).

 */

#include "includes.h"

// forward declarations.
idx_t da_getSimilarRows(da_csr_t *mat, idx_t rid, idx_t nsim, float eps,
        da_ivkv_t *hits, da_ivkv_t *i_cand, idx_t *i_marker, idx_t *ncands);

/**
 * Main entry point to KnnIdxJoin.
 */
void findNeighbors(params_t *params)
{

	ssize_t i, j, k, nneighbs;
	size_t rid, nsims, ncands, nnz;
	idx_t nrows, ncand, progressInd, pct;
	idx_t *marker=NULL;
	da_ivkv_t *hits=NULL, *cand=NULL;
	da_csr_t *docs, *neighbors=NULL;

	docs    = params->docs;
	nrows   = docs->nrows;  // num rows
	ncands  = 0; // number of considered candidates (computed similarities)
	nsims   = 0; // number of similar documents found

	/** Pre-process input matrix: remove empty columns, ensure sorted column ids, scale by IDF **/

    /* compact the column space */
    da_csr_CompactColumns(docs);
    if(params->verbosity > 0)
        printf("Docs matrix: " PRNT_IDXTYPE " rows, " PRNT_IDXTYPE " cols, "
            PRNT_PTRTYPE " nnz\n", docs->nrows, docs->ncols, docs->rowptr[docs->nrows]);

    /* sort the column space */
    da_csr_SortIndices(docs, DA_ROW);

    /* scale term values */
    if(params->verbosity > 0)
        printf("   Scaling input matrix.\n");
    da_csr_Scale(docs);


	timer_start(params->timer_3); /* overall knn graph construction time */

    /* normalize docs rows */
    da_csr_Normalize(docs, DA_ROW, 2);

    /* create inverted index - column version of the matrix */
	timer_start(params->timer_7); /* indexing time */
	da_csr_CreateIndex(docs, DA_COL);
	timer_stop(params->timer_7); /* indexing time */

	/* allocate memory for the search */
    timer_start(params->timer_5); /* memory allocation time */
    hits   = da_ivkvsmalloc(nrows, (da_ivkv_t) {0, 0.0}, "findNeighbors: hits"); /* empty list of key-value structures */
    cand   = da_ivkvsmalloc(nrows, (da_ivkv_t) {0, 0.0}, "findNeighbors: cand"); /* empty list of key-value structures */
    marker = da_ismalloc(nrows, -1, "findNeighbors: marker");  /* array of all -1 values */

    neighbors = da_csr_Create();
    neighbors->nrows = neighbors->ncols = nrows;
    nnz = params->k * docs->nrows; /* max number of neighbors */
    neighbors->rowptr = da_pmalloc(nrows + 1, "simSearchSetup: neighbors->rowptr");
    neighbors->rowind = da_imalloc(nnz, "simSearchSetup: neighbors->rowind");
    neighbors->rowval = da_vmalloc(nnz, "simSearchSetup: neighbors->rowval");
    neighbors->rowptr[0] = 0;
    timer_stop(params->timer_5); /* memory allocation time */

    /* set up progress indicator */
    da_progress_init_steps(pct, progressInd, nrows, 10);
	if(params->verbosity > 0)
		printf("Progress Indicator: ");

	/* execute search */
	for(nsims=0, i=0; i < nrows; i++){
		k = da_getSimilarRows(docs, i, params->k, params->epsilon, hits, cand, marker, &ncand);
		ncands += ncand;

		/* transfer candidates to output structure */
		for(j=0; j < k; j++){
	        neighbors->rowind[nsims] = hits[j].key;
	        neighbors->rowval[nsims] = hits[j].val;
	        nsims++;
		}
        neighbors->rowptr[i+1] = nsims;

		/* update progress indicator */
		if ( params->verbosity > 0 && i % progressInd == 0 ){
            da_progress_advance_steps(pct, 10);
		}
	}
	if(params->verbosity > 0){
            da_progress_finalize_steps(pct, 10);
	    printf("\n");
	}
	timer_stop(params->timer_3); // find neighbors time

    printf("Number of computed similarities: %zu\n", ncands);
    printf("Number of neighbors: %zu\n", nsims);

	/* write ouptut */
	if(params->oFile){
	    da_csr_Write(neighbors, params->oFile, DA_FMT_CSR, 1, 1);
	    printf("Wrote output to %s\n", params->oFile);
	}

	/* free memory */
	da_csr_Free(&neighbors);
	da_free((void**)&hits, &cand, &marker, LTERM);
}


/**
 * Find similar rows in the matrix -  this version of the function reports
 * the number of candidates/dot products that were considered in the search.
 * \param mat The CSR matrix we're searching in
 * \param rid Row we're looking for neighbors for
 * \param nsim Number of similar pairs to get (-1 to get all)
 * \param eps Minimum similarity between query and neighbors
 * \param hits Array or length mat->nrows to hold values for possible matches and result
 * \param i_cand Optional key-value array of length mat->nrows to store and sort candidates
 * \param i_marker Optional marker array of length mat->nrows to mark candidates
 * \param ncands Reference to int variable to hold number of candidates
 *
 * \return Number of similar pairs found
 */
idx_t da_getSimilarRows(da_csr_t *mat, idx_t rid, idx_t nsim, float eps,
        da_ivkv_t *hits, da_ivkv_t *i_cand, idx_t *i_marker, idx_t *ncands)
{
	ssize_t i, ii, j, k, qsz;
	idx_t nrows, ncols, ncand;
	ptr_t *colptr;
	idx_t *colind, *qind, *marker;
	val_t *colval, *qval;
	da_ivkv_t *cand;

	nrows  = mat->nrows;   /* number of rows */
	ncols  = mat->ncols;   /* number of columns */
	colptr = mat->colptr;  /* column pointers (where each column starts and ends in colind and colptr */
	colind = mat->colind;  /* column indices (document/row ids) */
	colval = mat->colval;  /* column values */
	qsz    = mat->rowptr[rid+1] - mat->rowptr[rid]; /* number of values in query row */
	qind   = mat->rowind + mat->rowptr[rid];        /* where indices (feature/column ids) for the query row start in the CSR structure */
	qval   = mat->rowval + mat->rowptr[rid];        /* where values for the query row start in the CSR structure */

    if (qsz == 0){
        return 0;
    }

	marker = (i_marker ? i_marker : da_ismalloc(nrows, -1, "da_csr_GetSimilarSmallerRows: marker"));
	cand   = (i_cand   ? i_cand   : da_ivkvmalloc(nrows, "da_csr_GetSimilarSmallerRows: cand"));

    for (ncand=0, ii=0; ii<qsz; ii++) {
        i = qind[ii];
        if (i < ncols) {
            for (j=colptr[i]; j<colptr[i+1]; j++) {
                k = colind[j];
                if(k == rid)
                    continue;
                if (marker[k] == -1) {
                    cand[ncand].key = k;
                    cand[ncand].val = 0;
                    marker[k]       = ncand++;
                }
                cand[marker[k]].val += colval[j] * qval[ii];
            }
        }
    }

	*ncands = ncand; /* number of candidates/computed similarities for this query object */

	/* clear markers */
	for (j=0, i=0; i<ncand; i++)
        marker[cand[i].key] = -1;

	if (nsim == -1 || nsim >= ncand) {
		nsim = ncand;
	}
	else {
		nsim = da_min(nsim, ncand);
		/* use select algorithm to get top k items */
		da_ivkvkselectd(ncand, nsim, cand);
	}
	/* filter out items below similarity threshold eps */
	for(k=0, i=0; i < nsim; ++i){
	    if(cand[i].val >= eps){
            hits[k].key = cand[i].key;
            hits[k].val = cand[i].val;
            k++;
	    }
	}

	if (i_marker == NULL)
		da_free((void **)&marker, LTERM);
	if (i_cand == NULL)
		da_free((void **)&cand, LTERM);

	return k;
}


