/*
 ============================================================================
 Name        : findsim.c
 Author      : David C. Anastasiu
 Version     :
 Copyright   : David C. Anastasiu
 Description : Find $k$ similar neighbors with at least $\epsilon$ similarity
 ============================================================================
 */

/*!
 \file  main.c
 \brief This file is the entry point for apss's various components

 \author David C. Anastasiu
 */

#include "includes.h"


/***********************************************/
/*! This is the entry point for the program    */
/***********************************************/
int main(int argc, char *argv[]) {
    params_t *params;

    params = (params_t *)da_malloc(sizeof(params_t), "main: params");

    cmdline_parse(params, argc, argv);

    if(params->verbosity > 0){
        printf("********************************************************************************\n");
        printf("%s (%d.%d.%d), vInfo: [%s]\n", PROGRAM_NAME, VER_MAJOR, VER_MINOR,
                VER_SUBMINOR, VER_COMMENT);
        printf("mode: %s, ", da_getStringKey(mode_options, params->mode));
        printf("iFile: %s, ", params->iFile);
        printf("oFile: %s, ", params->oFile ? params->oFile : "NULL");
        if(params->mode == MODE_TESTEQUAL) {
            printf("fldelta: %g", params->fldelta);
        }
        printf("k: %d, eps: %.2f", params->k, params->epsilon);
        printf("\n********************************************************************************\n");
        fflush(stdout);
    }

    timer_start(params->timer_global);

    // read input data
    readInputData(params);

    switch(params->mode){

    case MODE_IDXJOIN:
        findNeighbors(params);
        break;

    case MODE_TESTEQUAL:
        da_testMatricesEqual(params);
        break;

    case MODE_INFO:
        da_matrixInfo(params);
        break;

    case MODE_IO:
        da_matrixIo(params);
        break;

    case MODE_RECALL:
        da_testRecall(params);
        break;

    default:
        da_errexit("Invalid mode.");
        break;
    }


    // similarity search complete.
    if(params->verbosity > 0){
        timer_stop(params->timer_global);

        printf("TIMES:\n");
        da_printTimerLong("\t Similarity search: ", params->timer_3);
        da_printTimerLong("\t Total time: ", params->timer_global);

        printf(
                "********************************************************************************\n");
    }

    freeParams(&params);
    exit(EXIT_SUCCESS);
}

/**
 * Read input data
 */
void readInputData(params_t *params){
    da_csr_t *docs;

    if(!params->iFile){
        return;
    }

    params->fmtRead = da_getFileFormat(params->iFile, params->fmtRead);
    if(params->fmtRead < 1)
        da_errexit("Invalid input format.\n");
    docs = da_csr_Read(params->iFile, params->fmtRead, params->readVals, params->readNum);
    assert(docs->rowptr || docs->colptr);
    params->docs = docs;
}

/**
 * Test two matrices are equal. Values tested up to params->fldelta precision.
 */
void da_testMatricesEqual(params_t *params){
    da_csr_t *docs=NULL, *docs2=NULL;

    docs = params->docs;

    // test equality of two sparse matrices & print out differences
    docs2 = da_csr_Read(params->oFile, da_getFileFormat(params->oFile, params->fmtWrite),
            params->readVals, params->readNum);
    printf("Comparing %s (A[" PRNT_IDXTYPE "," PRNT_IDXTYPE "," PRNT_PTRTYPE "]) and "
            "%s (B[" PRNT_IDXTYPE "," PRNT_IDXTYPE "," PRNT_PTRTYPE "]).\n\n",
            params->iFile, docs->nrows, docs->ncols, docs->rowptr[docs->nrows],
            params->oFile, docs2->nrows, docs2->ncols, docs2->rowptr[docs2->nrows]);
    da_csrCompare(docs, docs2, params->fldelta, 1, 1);
    da_csr_Free(&docs2);
    freeParams(&params);
    exit(EXIT_SUCCESS);
}



/**
 * Test recall of knng solution.
 */
void da_testRecall(params_t *params){
    da_csr_t *docs=NULL, *docs2=NULL;

    docs = params->docs;

    docs2 = da_csr_Read(params->oFile, da_getFileFormat(params->oFile, params->fmtWrite),
            params->readVals, params->readNum);
    printf("Usage: findsim recall <true_results> <test_results>\n"
            "Use -verb 3 for additional information. Neighbors will be marked with:\n"
            "\t* neighbors that were missed with same value as the min values\n"
            "\t+ neighbors that were reported that are not in the true neighborhood\n"
            "\t- neighbors that were not reported\n\n");
    printf("True result matrix: %s (A[" PRNT_IDXTYPE "," PRNT_IDXTYPE "," PRNT_PTRTYPE "])\n"
            "Test result matrix: %s (B[" PRNT_IDXTYPE "," PRNT_IDXTYPE "," PRNT_PTRTYPE "]).\n\n",
            params->iFile, docs->nrows, docs->ncols, docs->rowptr[docs->nrows],
            params->oFile, docs2->nrows, docs2->ncols, docs2->rowptr[docs2->nrows]);
    verify_knng_results(docs2, docs, params->k, params->verbosity);
    da_csr_Free(&docs2);
    freeParams(&params);
    exit(EXIT_SUCCESS);
}

/**
 * Display information about a sparse matrix
 */
void da_matrixInfo(params_t *params){
    da_csr_t *docs = params->docs;
    size_t i, j, l, x1, x2, min, max;

    // identify format, read matrix, and print out information about it - nrows, ncols, nnz
    printf("%s: " PRNT_IDXTYPE " rows, " PRNT_IDXTYPE " cols, " PRNT_PTRTYPE " nnzs, %g density, ",
            params->iFile, docs->nrows, docs->ncols, docs->rowptr[docs->nrows],
            docs->rowptr[docs->nrows] / ((double) docs->nrows * docs->ncols)
    );

    da_csr_CompactColumns(docs);
    printf(PRNT_IDXTYPE " non-empty cols.\n", docs->ncols);

    if(params->stats){
        if(!docs->colptr)
            da_csr_CreateIndex(docs, DA_COL);
        for(x1=0, x2=0, min=INT_MAX, max=0, i=0; i < docs->nrows; ++i){
            l = docs->rowptr[i+1] - docs->rowptr[i];
            x1 += l;
            x2 += l*l;
            if(l > max)
                max = l;
            if(l < min)
                min = l;
        }
        printf("Row nnz stats: min %zu, max %zu mean %.2f, stdev %.2f.\n", min, max,
                (double)x1/(double)docs->nrows,
                sqrt( (double)x2*(double)docs->nrows - x1*x1 ) / (double)docs->nrows );
        for(x1=0, x2=0, i=0; i < docs->ncols; ++i){
            l = docs->colptr[i+1] - docs->colptr[i];
            x1 += l;
            x2 += l*l;
            if(l > max)
                max = l;
            if(l < min)
                min = l;
        }
        printf("Col nnz stats: min %zu, max %zu mean %.2f, stdev %.2f.\n", min, max,
                (double)x1/(double)docs->ncols,
                sqrt( (double)x2*(double)docs->ncols - x1*x1 ) / (double)docs->ncols );
    }



    printf("\n");

    freeParams(&params);
    exit(EXIT_SUCCESS);
}


/**
 * Transform input matrix to some other format
 * Data pre-processing invoked before transforming.
 */
void da_matrixIo(params_t *params){
    da_csr_t *docs = params->docs;

    if(params->oFile){
        params->fmtWrite = da_getFileFormat(params->oFile, params->fmtWrite);
        if(params->fmtWrite < 1)
            da_errexit("Invalid output format.\n");
    }

    if(params->verbosity > 0)
        printf("Transforming %s (A[" PRNT_IDXTYPE "," PRNT_IDXTYPE "," PRNT_PTRTYPE "]) from "
            "%s to %s, saving to %s ...\n",
            params->iFile, docs->nrows, docs->ncols, docs->rowptr[docs->nrows],
            da_getStringKey(fmt_options, params->fmtRead),
            da_getStringKey(fmt_options, params->fmtWrite), params->oFile);

    if(!docs->rowptr)
        da_csr_CreateIndex(docs, DA_ROW);

    da_csr_Write(docs, params->oFile, params->fmtWrite, params->writeVals, params->writeNum);
    if(params->verbosity > 0)
        printf("Done.\n");

    freeParams(&params);
    exit(EXIT_SUCCESS);
}


/**
 * Free memory from the params structure
 */
void freeParams(params_t** params){
    da_csr_FreeAll(&(*params)->docs, &(*params)->neighbors, LTERM);
    da_free((void**)&(*params)->iFile, &(*params)->oFile, &(*params)->vFile,
            &(*params)->filename, LTERM);

    da_free((void**)params, LTERM);
}
