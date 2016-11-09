/*!
 \file  dynamic_index_speedup_version_2.cc

 This file contains a faster implementation of the all pair simialrities.
 Basic idea is to leverage the dynamic index. 
    - First and biggest advantage of doing that is it reduces computation by half.
 
 The algorithm is based on exploiting threshold during indexing technique as
 proposed by authors of research paper:
 
 R. J. Bayardo, Y. Ma, and R. Srikant, “Scaling up all pairs similarity search,” 
 in Proceedings of the 16th International Conference on World Wide Web, ser. WWW 
 ’07. New York, NY, USA: ACM, 2007, pp. 131–140”

\author Sanisha Rehan
 */

#include "includes.h"

#include <functional>
#include <set>
#include <unordered_map>
#include <vector>
#include <iostream>

using namespace std;

//
// This function computes the prefix cosine similarity of two given vectors.
//
float getPrefixDotProduct(int vector_id, int vector_with_prefix_id,
        const std::vector<int>& vector_prefix_indices, da_csr_t* docs) {
    float result = 0;
    for (int i = docs->rowptr[vector_with_prefix_id], j = docs->rowptr[vector_id];
            i <= vector_prefix_indices[vector_with_prefix_id] && j < docs->rowptr[vector_id + 1]; ) {
        if (docs->rowind[i] == docs->rowind[j]) {
            result += docs->rowval[i++] * docs->rowval[j++];
        } else if (docs->rowind[i] < docs->rowind[j]) {
            ++i;
        } else {
            ++j;
        }
    }

    return result;
}

// This function is supposed to find similarities with the index 
// generated till now. This partial index would include data corresponfing
// to indices till doc_id.
void UpdateSimilarities(const int doc_id,
    const std::vector<std::vector<std::pair<int, float>>>& dynamic_index,
    const float threshold, 
    const std::vector<int>& vector_prefix_indices,
    da_csr_t* docs, 
    std::vector<std::set<std::pair<float, int>, 
                         std::greater<std::pair<float, int>>>>* similarities) {

    // Store all the dot products.
    std::vector<float> dot_products;
    dot_products.resize(doc_id, 0);

    // Calculate dot products.
    for(int i=docs->rowptr[doc_id]; i < docs->rowptr[doc_id + 1]; ++i) {
        for (const auto& doc_id_weight_pair: dynamic_index[docs->rowind[i]]) {
            dot_products[doc_id_weight_pair.first] += 
                docs->rowval[i] * doc_id_weight_pair.second;
        }
    }

    // If any dot product is greater than threshold, store that.
    for(int i=0; i < dot_products.size(); ++i) {
        float new_product = 0.0;
        if (dot_products[i] > 0) {
            new_product = dot_products[i] + getPrefixDotProduct(doc_id, i, vector_prefix_indices, docs);
        }
        if (new_product >= threshold) {
            (*similarities)[doc_id].insert({new_product, i});
            (*similarities)[i].insert({new_product, doc_id});
        }
    }
}
 

/**
 * Main entry point to Faster Dynamic Find Neighbors.
 */
void dynamic_findNeighbors_version_2(params_t *params)
{
	size_t nsims, nnz;
    idx_t nrows, progress_ind, pct;
	da_csr_t *docs, *neighbors=NULL;

	docs    = params->docs;
	nrows   = docs->nrows;  // num rows
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

    // Create the index so that we can find the max value for each column.
	da_csr_CreateIndex(docs, DA_COL);

    /*Initialize the output data structure.*/
    neighbors = da_csr_Create();
    neighbors->nrows = neighbors->ncols = nrows;
    nnz = params->k * docs->nrows; /* max number of neighbors */
    neighbors->rowptr = da_pmalloc(nrows + 1, "simSearchSetup: neighbors->rowptr");
    neighbors->rowind = da_imalloc(nnz, "simSearchSetup: neighbors->rowind");
    neighbors->rowval = da_vmalloc(nnz, "simSearchSetup: neighbors->rowval");
    neighbors->rowptr[0] = 0;
 
    // Initialize progress indicator
	da_progress_init_steps(pct, progress_ind, nrows, 10);
    if (params->verbosity > 0) {
        printf("Progress Indicator: ");
    }
    
    // Find maximum values for each feature/column.
    std::vector<float> col_max;
    col_max.resize(docs->ncols, 0);
    for (int i=0; i < docs->ncols; ++i) {
        for (int j = docs->colptr[i]; j < docs->colptr[i+1]; ++j) {
            col_max[i] = max(col_max[i], docs->colval[j]);
        }
    }

    printf("Rows=%d and Cols=%d\n", docs->nrows, docs->ncols);

    // Data structure for storing neighbors for each vector with 
    // similarity greater than threshold.
    // e.g: v1 -> [(.4, v3), ()]
    // We use set with greater so that we can find top K sorted by default.
    std::vector<std::set<std::pair<float, int>, 
                         std::greater<std::pair<float, int>>>> similarities;
    
    // Resize this vector to match the number of documents.
    similarities.resize(docs->nrows);
    
    // Data structure for storing the dynamic index.
    // e.g
    // I1 -> [(v1, .2), (v45, .5),....]
    // I2 -> [(v200, .7), (v14, .23),....]
    // ...
    std::vector<std::vector<std::pair<int, float>>> dynamic_index;
    
    // Resize the index to match the feature size.
    dynamic_index.resize(docs->ncols);
   
    // This vector would store the index(location in csr_doc) of the feature 
    // in each document, from where onwards we create the index.
    //
    // Therefore, all the candidate neighbors would have to take dot product
    // with this vector.
    std::vector<int> vector_prefix_indices;
    vector_prefix_indices.resize(docs->nrows);

    // This piece of code is supposed to calculate similarities and build index dynamically.
    for (int i=0; i < docs->nrows; ++i) {
        UpdateSimilarities(i, dynamic_index, params->epsilon, 
            vector_prefix_indices, docs, &similarities);
        float b = 0;
        for (int j = docs->rowptr[i]; j < docs->rowptr[i+1]; ++j) {
            b += col_max[docs->rowind[j]] * docs->rowval[j];
            if (b >= params->epsilon) {
                dynamic_index[docs->rowind[j]].push_back({i, docs->rowval[j]});     
            } else {
                vector_prefix_indices[i] = j;
            }
        }

        // Update progress indicator.
        if (params->verbosity > 0 && i % progress_ind == 0) {
            da_progress_advance_steps(pct, 10);
        }
    }

    // Go over all the similarities for each document vector and find out the
    // top K.
    nsims = 0;
    for (int i=0; i < similarities.size(); ++i) {
        int j = 0;
        for (const auto& dot_product_similarity : similarities[i]) {
            if (j >= params->k) {
                break;
            }
            neighbors->rowind[nsims] = dot_product_similarity.second;
            neighbors->rowval[nsims] = dot_product_similarity.first;
            ++j;
            ++nsims;
        }
        neighbors->rowptr[i+1] = nsims;
    }

    // Print progress indicator.
    if (params->verbosity > 0) {
        da_progress_finalize_steps(pct, 10);
        printf("\n");
    }

	timer_stop(params->timer_3); // find neighbors time

    /* Write ouptut */
	if(params->oFile){
	    da_csr_Write(neighbors, params->oFile, DA_FMT_CSR, 1, 1);
	    printf("Wrote output to %s\n", params->oFile);
	}
	da_csr_Free(&neighbors);
}
