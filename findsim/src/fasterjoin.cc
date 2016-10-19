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
#include <stdbool.h>
#include <set>
#include <vector>

using namespace std;

// forward declarations.
idx_t fast_getSimilarRows(da_csr_t *mat, idx_t rid, idx_t nsim, float eps,
        da_ivkv_t *hits, da_ivkv_t *i_cand, idx_t *i_marker, idx_t *ncands);

// Copied create Index function from da_csr and tailored for my needs.
void fast_CreateIndex(da_csr_t* const mat, const char what, 
        val_t* r_max, val_t* c_max, bool* ignore_rows)
{
	/* 'f' stands for forward, 'r' stands for reverse */
	ssize_t i, j, k, nf, nr;
	ptr_t *fptr, *rptr;
	idx_t *find, *rind;
	val_t *fval, *rval;

	switch (what) {
	case DA_COL:
		nf   = mat->nrows;
		fptr = mat->rowptr;
		find = mat->rowind;
		fval = mat->rowval;

		if (mat->colptr) da_free((void **)&mat->colptr, LTERM);
		if (mat->colind) da_free((void **)&mat->colind, LTERM);
		if (mat->colval) da_free((void **)&mat->colval, LTERM);

		nr   = mat->ncols;
		rptr = mat->colptr = da_pnmalloc(nr+1, "fast_CreateIndex: rptr");
		rind = mat->colind = da_imalloc(fptr[nf], "fast_CreateIndex: rind");
		rval = mat->colval = (fval ? da_vmalloc(fptr[nf], "fast_CreateIndex: rval") : NULL);
		break;
	case DA_ROW:
		nf   = mat->ncols;
		fptr = mat->colptr;
		find = mat->colind;
		fval = mat->colval;

		if (mat->rowptr) da_free((void **)&mat->rowptr, LTERM);
		if (mat->rowind) da_free((void **)&mat->rowind, LTERM);
		if (mat->rowval) da_free((void **)&mat->rowval, LTERM);

		nr   = mat->nrows;
		rptr = mat->rowptr = da_pnmalloc(nr+1, "fast_CreateIndex: rptr");
		rind = mat->rowind = da_imalloc(fptr[nf], "fast_CreateIndex: rind");
		rval = mat->rowval = (fval ? da_vmalloc(fptr[nf], "fast_CreateIndex: rval") : NULL);
		break;
	default:
		da_errexit( "Invalid index type of %d.\n", what);
		return;
	}


    // What is ptr for? It is for storing where would next row index or 
    // value would start from. Basically gives an idea about number of
    // elements stored in each row of the compressed matrix.
	for (i=0; i<nf; ++i) {
		for (j=fptr[i]; j<fptr[i+1]; ++j)
			rptr[find[j]]++;
	}
	CSRMAKE(i, nr, rptr);

    // Why do this 6*nr thing is not clear to me. Not going to invest more time also here.
	if (rptr[nr] > 6*nr) {
		for (i=0; i<nf; ++i) {
			for (j=fptr[i]; j<fptr[i+1]; ++j)
				rind[rptr[find[j]]++] = i;
		}
		CSRSHIFT(i, nr, rptr);

		if (fval) {
			for (i=0; i<nf; ++i) {
				for (j=fptr[i]; j<fptr[i+1]; ++j)
					rval[rptr[find[j]]++] = fval[j];
			}
			CSRSHIFT(i, nr, rptr);
		}
	}
	else {
		if (fval) {
			for (i=0; i<nf; ++i) {
				for (j=fptr[i]; j<fptr[i+1]; ++j) {
					k = find[j];
					rind[rptr[k]]   = i;
					rval[rptr[k]++] = fval[j];
				}
			}
		}
		else {
			for (i=0; i<nf; ++i) {
				for (j=fptr[i]; j<fptr[i+1]; ++j)
					rind[rptr[find[j]]++] = i;
			}
		}
		CSRSHIFT(i, nr, rptr);
	}
}

/**
 * Main entry point to KnnIdxJoin.
 */
void fast_findNeighbors(params_t *params)
{

	ssize_t i, j, k, nneighbs;
	size_t rid, nsims, ncands, nnz;
	idx_t nrows, ncand, progressInd, pct;
	idx_t *marker=NULL;
	da_ivkv_t *hits=NULL, *cand=NULL;


    // This is the main data structure.
    // da_csr_t. Compressed row storage.
    //
    // row_ptr stores the index from which data for a data vector starts.
    // col_ind stores the index of the dimension in the corresponding row_val. 
    // and as you guessed correctly row_val is the actual value stored in the sparse 2-D matrix.
    //
    // Example:
    // Matrix:
    //
    // 10 0 0 0 -2 0
    //  3 9 0 0  0 3
    //  0 7 8 7  0 0
    //  3 0 8 7  5 0
    //
    // vals = [10, -2, 3, 9, 3, 7, 8, 7, 3, 8, 7, 5]
    // indx = [ 0,  4, 0, 1, 5, 1, 2, 3, 0, 2, 3, 4]
    // ptrs = [0, 2, 5, 8, 12]
	da_csr_t *docs, *neighbors=NULL;

	docs    = params->docs;
	nrows   = docs->nrows;  // num rows
	ncands  = 0; // number of considered candidates (computed similarities)
	nsims   = 0; // number of similar documents found

	/** Pre-process input matrix: remove empty columns, ensure sorted column ids, scale by IDF **/

    // 1) Remove the empty columns. 
    // 2) Re-number columns.
    // 3) Sorts columns by decreasing order of frequency, i.e features which are most
    //    column are put in front.
    // Re-numbering is fine, because we only care about the row numbers. Columns we 
    // can tweak whatever way we want.
    da_csr_CompactColumns(docs);

    if(params->verbosity > 0) {
        printf("Docs matrix: " PRNT_IDXTYPE " rows, " PRNT_IDXTYPE " cols, "
            PRNT_PTRTYPE " nnz\n", docs->nrows, docs->ncols, docs->rowptr[docs->nrows]);
    }

    /* sort the column space */
    // Once this is done, all the columns would be sorted for each row.
    da_csr_SortIndices(docs, DA_ROW);

    /* scale term values */
    if(params->verbosity > 0)
        printf("Scaling input matrix.\n");

    // IDF(Inverse document frequency) scaling is required to reuduce the impact of words like "the", "this".
    //
    // While doing document matching, we should be discounting the columns which have the highest frequency.
    // Essentially, all frequency values would be scaled down as 
    // val[i] = val[i] * log (total_documents / total_occurence_of_word (total of that column across all vectors/documnets).
    //
    da_csr_Scale(docs);

	timer_start(params->timer_3); /* overall knn graph construction time */

    /* normalize docs rows */
    // Here all we do is divide each value in the row by
    // SQRT(sum of squares of all values in vector).
    da_csr_Normalize(docs, DA_ROW, 2);

    /* create inverted index - column version of the matrix */
	timer_start(params->timer_7); /* indexing time */

	da_csr_CreateIndex(docs, DA_COL);

    /*
    // Some variables for iteration.
    size_t ri, ci, rci, cri;

    // All the row maximums.
    val_t* r_max = da_vmalloc(docs->nrows, "Initialize row maximum values.");
    for (ri = 0; ri < nrows; ++ri) {
        val_t max = 0;
        for (rci = docs->rowptr[ri]; rci < docs->rowptr[ri+1]; ++rci) {
            if (docs->rowval[rci] > max) {
                max = docs->rowval[rci];
            }
        }
        r_max[ri] = max;
        //printf("row: %d, max_val: %f \n", ri, r_max[ri]);
    }
    da_vsorti(docs->nrows, r_max);
    for (ri = 0; ri < nrows; ++ri) {
        printf("row: %d, max_val: %f \n", ri, r_max[ri]);
    }
    
    // All the column maximums.
    val_t* c_max = da_vmalloc(docs->ncols, "Initialize col maximum values.");

    for (ci = 0; ci < docs->ncols; ++ci) {
        val_t max = 0;
        for (cri = docs->colptr[ci]; cri < docs->colptr[ci+1]; ++cri) {
            if (docs->colval[cri] > max) {
                max = docs->colval[cri];
            }
        }
        c_max[ci] = max;
        //printf("col: %d, max_val: %f \n", ci, c_max[ci]);
    }
    da_vsorti(docs->ncols, c_max);
    for (ci = 0; ci < docs->ncols; ++ci) {
        printf("col: %d, max_val: %f \n", ci, c_max[ci]);
    }

    // Stores whether we can ignore the document totally for doc similarity computations.
    bool* ignore_rows = (bool*) malloc(sizeof(bool*) * docs->nrows);
    for (ri = 0; ri < docs->nrows; ++ri) {
        val_t max_dot = 0;
        for (rci = docs->rowptr[ri]; rci < docs->rowptr[ri + 1]; ++rci) {
            max_dot += c_max[docs->rowind[rci]] * docs->rowval[rci];
        }
        printf("row: %d, max_dot: %f \n", ri, max_dot);
    }*/

	timer_stop(params->timer_7); /* indexing time */

    /* allocate memory for the search */
    timer_start(params->timer_5); /* memory allocation time */
    hits   = da_ivkvsmalloc(nrows, (da_ivkv_t) {0, 0.0}, "findNeighbors: hits"); /* empty list of key-value structures */
    cand   = da_ivkvsmalloc(nrows, (da_ivkv_t) {0, 0.0}, "findNeighbors: cand"); /* empty list of key-value structures */
    marker = da_ismalloc(nrows, -1, "findNeighbors: marker");  /* array of all -1 values */


    /// COME BACK HERE AND START READING FROM THIS POINT !!!!!
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

    // Accumulate the results for all together.
    std::vector<std::vector<float>> A;
    A.reserve(docs->nrows);
    for (int i=0; i < docs->nrows; ++i) {
        A[i].resize(docs->nrows, 0);
    }

    float temp = 0;
    for (int i=0; i < docs->ncols; ++i) {
        for (int j = docs->colptr[i]; j < docs->colptr[i+1]; ++j) {
            for (int k = j+1; k < docs->colptr[i+1]; ++k) {
                temp = docs->colval[j] * docs->colval[k];
                A[docs->colind[j]][docs->colind[k]] += temp;
            }
        }
    }

    for (int i=0; i < docs->nrows; ++i) {
        for (int j=0; j < i; ++j) {
            A[i][j] = A[j][i];
        }
    }

    nsims = 0;
    for (int i=0; i < docs->nrows; ++i) {
        std::set<std::pair<float, int>> topk;
        for (int j=0; j < docs->nrows; ++j) {
            if (i == j) { continue; }
            if (A[i][j] <= params->epsilon) { continue; }
            if (topk.size() < params->k) {
                topk.insert(std::pair<float, int>(A[i][j], j));
            } else {
                // TODO (sanisha): Add code here.
            }
        }

        std::set<std::pair<int, float>> reversed_topk;
        for (auto topk_ele : topk) {
            reversed_topk
                .insert(std::pair<int, float>(topk_ele.second, topk_ele.first));
        }

        for (auto topk_ele : reversed_topk) {
            neighbors->rowind[nsims] = topk_ele.first;
            neighbors->rowval[nsims] = topk_ele.second;
            ++nsims;
        }
        neighbors->rowptr[i+1] = nsims;
		if ( params->verbosity > 0 && i % progressInd == 0 ){
            da_progress_advance_steps(pct, 10);
		}
    }

	/* // execute search 
	for(nsims=0, i=0; i < nrows; i++){
		k = fast_getSimilarRows(docs, i, params->k, params->epsilon, hits, cand, marker, &ncand);
		ncands += ncand;

		// transfer candidates to output structure.
		for(j=0; j < k; j++){
	        neighbors->rowind[nsims] = hits[j].key;
	        neighbors->rowval[nsims] = hits[j].val;
	        nsims++;
		}
        neighbors->rowptr[i+1] = nsims;

		// update progress indicator
		if ( params->verbosity > 0 && i % progressInd == 0 ){
            da_progress_advance_steps(pct, 10);
		}
	}*/
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
idx_t fast_getSimilarRows(da_csr_t *mat, idx_t rid, idx_t nsim, float eps,
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
                    marker[k] = ncand++;
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


