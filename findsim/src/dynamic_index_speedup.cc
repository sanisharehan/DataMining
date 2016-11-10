/*!
 \file  dynamic_index_speedup.c

 \brief This file contains Dynamic Inverted Index algorithm related functions.
 This algorithm builds the inverted indices dynamically and incrementally, improving
 the memory footprint and locality of the algorithm. Also, we leverage the 
 commutative property of cosine similarity function which reduces the similarity
 calculations by half, thus further improving the overall speed. The final results
 are sorted based on their respective similarities and only those results are retained
 that should be part of the output (top-$k$ results with at least $\epsilon$ similarity).

 \author Sanisha Rehan
 */

#include "includes.h"

#include <functional>
#include <unordered_map>
#include <set>
#include <vector>
#include <iostream>


using namespace std;

// This function is supposed to find similarities with the index 
// generated till now. This partial index would include data corresponfing
// to indices till doc_id.
void UpdateSimilarities(const int doc_id,
    const std::vector<std::vector<std::pair<int, float>>>& dynamic_index,
    const float threshold, da_csr_t* docs, 
    std::vector<std::vector<std::pair<float, int>>>* similarities) {
    
    // Store all the dot products.
    std::vector<float> dot_products;
    dot_products.resize(doc_id + 1, 0);

    // Calculate dot products.
    for(int i=docs->rowptr[doc_id]; i < docs->rowptr[doc_id + 1]; ++i) {
        for (const auto& doc_id_weight_pair: dynamic_index[docs->rowind[i]]) {
            dot_products[doc_id_weight_pair.first] += 
                docs->rowval[i] * doc_id_weight_pair.second;
        }
    }

    // If any dot product is greater than threshold, store that.
    for(size_t i=0; i < dot_products.size(); ++i) {
        if (dot_products[i] >= threshold) {
            (*similarities)[doc_id].push_back(std::pair<float, int>(dot_products[i], i));
            (*similarities)[i].push_back(std::pair<float, int>(dot_products[i], doc_id));
        }
    }
}
 

/**
 * Main entry point to Dynamic Inverted Index Algorithm.
 */
void dynamic_findNeighbors(params_t *params)
{

	size_t nsims, nnz;
	idx_t nrows, progress_ind, pct;
	da_csr_t *docs, *neighbors=NULL;

	docs    = params->docs;
	nrows   = docs->nrows;  // num rows
	nsims   = 0;            // number of similar documents found

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

    // Initialize the output data structure.
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

    // Data structure for storing neighbors for each vector with 
    // similarity greater than threshold.
    // e.g: Row0: v1 -> [(.4, v3), ()]
    //      Row1: v2 -> [(.7, v5), (.8, v6)]
    //      ...
    std::vector<std::vector<std::pair<float, int>>> similarities;

    // Resize this vector to match the number of documents.
    similarities.resize(docs->nrows);
    
    // Data structure for storing the dynamic index. Its a vector of vectors.
    // Each row indicates the inverted index for an attribute.
    // e.g
    // Row0: I1 -> [(v1, .2), (v45, .5),....]
    // Row1: I2 -> [(v200, .7), (v14, .23),....]
    // ...
    std::vector<std::vector<std::pair<int, float>>> dynamic_index;
    
    // Resize the index to match the feature size i.e. maximum number of features.
    dynamic_index.resize(docs->ncols);
   
    // This piece of code is supposed to calculate
    // similarities and build index dynamically.
    for (int i=0; i < docs->nrows; ++i) {
        UpdateSimilarities(i, dynamic_index, params->epsilon, docs, &similarities);
        for (int j = docs->rowptr[i]; j < docs->rowptr[i+1]; ++j) {
            dynamic_index[docs->rowind[j]].push_back({i, docs->rowval[j]});     
        }

        // Update progress indicator.
        if (params->verbosity > 0 && i % progress_ind == 0) {
            da_progress_advance_steps(pct, 10);
        }

    }

    // Go over all the similarities for each document vector and find out the
    // top K.
    nsims = 0;
    for (size_t i=0; i < similarities.size(); ++i) {
        int j = 0;
        // This set has top k similarities in ascending order.
        std::set<std::pair<float, int>> topk_ele;
        for (const auto& dot_product_similarity : similarities[i]) {
            if (topk_ele.size() < params->k) {
                topk_ele.insert(std::pair<float, int>(dot_product_similarity.first, dot_product_similarity.second));
            } else if (topk_ele.size() == params->k) {
                // Check if the given similarity value is more than the minimum
                // value present in the set or not.
                std::set<std::pair<float, int>>::iterator topk_it;
                topk_it = topk_ele.begin();
                if (topk_it->first < dot_product_similarity.first) {
                    topk_ele.erase(topk_it);
                    topk_ele.insert(std::pair<float, int>(dot_product_similarity.first, dot_product_similarity.second));
                }
            }
        }
            
        // Create a set to get top k similarities in descending order.
        std::set<std::pair<float, int>, std::greater<std::pair<float, int>>> topk_ele_desc;
        topk_ele_desc.insert(topk_ele.begin(), topk_ele.end());
            
        for (const auto &top_ele : topk_ele_desc) {
            neighbors->rowind[nsims] = top_ele.second;
            neighbors->rowval[nsims] = top_ele.first;
            ++nsims;
        }
        neighbors->rowptr[i+1] = nsims;
    }

    // Print progress indicator
    if (params->verbosity > 0) {
        da_progress_finalize_steps(pct, 10);
        printf("\n");
    }

	timer_stop(params->timer_3); // find neighbors time

    printf("Number of neighbors: %zu\n", nsims);

    /* write ouptut */
	if(params->oFile){
	    da_csr_Write(neighbors, params->oFile, DA_FMT_CSR, 1, 1);
	    printf("Wrote output to %s\n", params->oFile);
	}
	da_csr_Free(&neighbors);
}
