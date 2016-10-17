/*!
 \file  csr.c
 \brief Functions for dealing with csr structures

Some functions initially ported from George Karypis' GKlib library by David C. Anastasiu,
with permission, in Aug 2013, then modified.

 \author David C. Anastasiu
 */
#include "includes.h"


/*************************************************************************/
/*! Allocate memory for a CSR matrix and initializes it
    \returns the allocated matrix. The various fields are set to NULL.
 */
/**************************************************************************/
da_csr_t* da_csr_Create(void)
{
	da_csr_t *mat;

	mat = (da_csr_t *)da_malloc(sizeof(da_csr_t), "da_csr_Create: mat");

	da_csr_Init(mat);

	return mat;
}


da_csr_t* da_csr_Alloc(
        da_csr_t* const mat,
        const idx_t nrows,
        const idx_t ncols,
        const ptr_t nnz,
        const uint what,
        const bool values)
{
    mat->nrows = nrows;
    mat->ncols = ncols;

    if((what & DA_ROW)){
        if (mat->rowptr) da_free((void **)&mat->rowptr, LTERM);
        if (mat->rowind) da_free((void **)&mat->rowind, LTERM);
        if (mat->rowval) da_free((void **)&mat->rowval, LTERM);

        mat->rowptr = da_pnmalloc_a(nrows+1, "da_csr_CreateIndex: rptr");
        mat->rowind = da_imalloc_a(nnz, "da_csr_CreateIndex: rind");
        mat->rowval = (values ? da_vmalloc_a(nnz, "da_csr_CreateIndex: rval") : NULL);
    }

    if((what & DA_COL)){
        if (mat->colptr) da_free((void **)&mat->colptr, LTERM);
        if (mat->colind) da_free((void **)&mat->colind, LTERM);
        if (mat->colval) da_free((void **)&mat->colval, LTERM);

        mat->colptr = da_pnmalloc_a(ncols+1, "da_csr_CreateIndex: rptr");
        mat->colind = da_imalloc_a(nnz, "da_csr_CreateIndex: rind");
        mat->colval = (values ? da_vmalloc_a(nnz, "da_csr_CreateIndex: rval") : NULL);
    }

    return mat;
}


/*************************************************************************/
/*! Initializes the matrix
    \param mat is the matrix to be initialized.
 */
/*************************************************************************/
void da_csr_Init(da_csr_t* const mat)
{
	memset(mat, 0, sizeof(da_csr_t));
	mat->nrows = mat->ncols = -1;
}


/*************************************************************************/
/*! Frees all the memory allocated for matrix.
    \param mat is the matrix to be freed.
 */
/*************************************************************************/
void da_csr_Free(da_csr_t** mat)
{
	if (*mat == NULL)
		return;
	da_csr_FreeContents(*mat);
	da_free((void **)mat, LTERM);
}


/*************************************************************************/
/*! Frees a variable sized list of matrix pointers. Last item in the list must be LTERM.
    \param ptr1 is the first matrix to be freed,
    \param ... are additional matrices to be freed.
 */
/*************************************************************************/
void da_csr_FreeAll(da_csr_t** ptr1, ...)
{
	va_list plist;
	void **ptr;

	if (*ptr1 != NULL)
		da_csr_Free(ptr1);

	va_start(plist, ptr1);
	while ((ptr = va_arg(plist, void **)) != LTERM) {
		if (*ptr != NULL) {
			da_csr_Free((da_csr_t **)ptr);
		}
	}
	va_end(plist);
}


/*************************************************************************/
/*! Frees the memory allocated for the matrix's fields for the given type and
    sets them to NULL.
    \param mat is the matrix whose partial contents will be freed,
    \param type is the internal representation to be freed: row (DA_ROW),
    		or column (DA_COL).
 */
/*************************************************************************/
void da_csr_FreeBase(da_csr_t* const mat, const char type)
{
	if(type & DA_ROW)
		//drop the row index
		da_free((void **)&mat->rowptr, &mat->rowind, &mat->rowval, LTERM);

	if(type & DA_COL)
		//drop the column index
		da_free((void **)&mat->colptr, &mat->colind, &mat->colval, LTERM);

}


/*************************************************************************/
/*! Create missing indexes for the matrix such that both indexes exist.
    \param mat is the matrix whose bases need loading.
 */
/*************************************************************************/
void da_csr_LoadBases(da_csr_t* const csr)
{
	if(csr->rowptr && csr->colptr) return;
	if(!csr->rowptr && !csr->colptr) return;
	if(csr->rowptr){
		da_csr_CreateIndex(csr, DA_COL);
		// sort column indices for input matrices
		da_csr_SortIndices(csr, DA_COL);
	} else {
		da_csr_CreateIndex(csr, DA_ROW);
		// sort column indices for input matrices
		da_csr_SortIndices(csr, DA_ROW);
	}
}



/*************************************************************************/
/*! Frees only the memory allocated for the matrix's different fields and
    sets them to NULL.
    \param mat is the matrix whose contents will be freed.
 */
/*************************************************************************/
void da_csr_FreeContents(da_csr_t* const mat)
{
	da_free((void *)&mat->rowptr, &mat->rowind, &mat->rowval,
			&mat->colptr, &mat->colind, &mat->colval,
			&mat->rnorms, &mat->cnorms,
			LTERM);
}


/*************************************************************************/
/*! Returns a copy of a matrix.
    \param mat is the matrix to be duplicated.
    \returns the newly created copy of the matrix.
 */
/**************************************************************************/
da_csr_t* da_csr_Copy(const da_csr_t* const mat)
{
	da_csr_t *nmat;

	nmat = da_csr_Create();

	nmat->nrows  = mat->nrows;
	nmat->ncols  = mat->ncols;

	/* copy the row structure */
	if (mat->rowptr)
		nmat->rowptr = da_pcopy(mat->nrows+1, mat->rowptr,
				da_pmalloc(mat->nrows+1, "da_csr_Dup: rowptr"));
	if (mat->rnorms)
		nmat->rnorms = da_vcopy(mat->nrows, mat->rnorms,
				da_vmalloc(mat->nrows, "da_csr_Dup: rnorms"));
	if (mat->rowind)
		nmat->rowind = da_icopy(mat->rowptr[mat->nrows], mat->rowind,
				da_imalloc(mat->rowptr[mat->nrows], "da_csr_Dup: rowind"));
	if (mat->rowval)
		nmat->rowval = da_vcopy(mat->rowptr[mat->nrows], mat->rowval,
				da_vmalloc(mat->rowptr[mat->nrows], "da_csr_Dup: rowval"));

	/* copy the col structure */
	if (mat->colptr)
		nmat->colptr = da_pcopy(mat->ncols+1, mat->colptr,
				da_pmalloc(mat->ncols+1, "da_csr_Dup: colptr"));
	if (mat->cnorms)
		nmat->cnorms = da_vcopy(mat->ncols, mat->cnorms,
				da_vmalloc(mat->ncols, "da_csr_Dup: cnorms"));
	if (mat->colind)
		nmat->colind = da_icopy(mat->colptr[mat->ncols], mat->colind,
				da_imalloc(mat->colptr[mat->ncols], "da_csr_Dup: colind"));
	if (mat->colval)
		nmat->colval = da_vcopy(mat->colptr[mat->ncols], mat->colval,
				da_vmalloc(mat->colptr[mat->ncols], "da_csr_Dup: colval"));

	return nmat;
}


da_csr_t* da_csr_Copy_a(
        const da_csr_t* const mat)
{
    da_csr_t *nmat;

    nmat = da_csr_Create();

    nmat->nrows  = mat->nrows;
    nmat->ncols  = mat->ncols;

    /* copy the row structure */
    if (mat->rowptr)
        nmat->rowptr = da_pcopy(mat->nrows+1, mat->rowptr,
                da_pmalloc_a(mat->nrows+1, "da_csr_Copy: rowptr"));
    if (mat->rowind)
        nmat->rowind = da_icopy(mat->rowptr[mat->nrows], mat->rowind,
                da_imalloc_a(mat->rowptr[mat->nrows], "da_csr_Copy: rowind"));
    if (mat->rowval)
        nmat->rowval = da_vcopy(mat->rowptr[mat->nrows], mat->rowval,
                da_vmalloc_a(mat->rowptr[mat->nrows], "da_csr_Copy: rowval"));
    if (mat->rnorms)
        nmat->rnorms = da_vcopy(mat->nrows, mat->rnorms,
                da_vmalloc_a(mat->nrows, "da_csr_Copy: rnorms"));

    /* copy the col structure */
    if (mat->colptr)
        nmat->colptr = da_pcopy(mat->ncols+1, mat->colptr,
                da_pmalloc_a(mat->ncols+1, "da_csr_Copy: colptr"));
    if (mat->colind)
        nmat->colind = da_icopy(mat->colptr[mat->ncols], mat->colind,
                da_imalloc_a(mat->colptr[mat->ncols], "da_csr_Copy: colind"));
    if (mat->colval)
        nmat->colval = da_vcopy(mat->colptr[mat->ncols], mat->colval,
                da_vmalloc_a(mat->colptr[mat->ncols], "da_csr_Copy: colval"));
    if (mat->cnorms)
        nmat->cnorms = da_vcopy(mat->ncols, mat->cnorms,
                da_vmalloc_a(mat->ncols, "da_csr_Copy: cnorms"));

    return nmat;
}


void da_csr_Grow_a(
        da_csr_t* const mat,
        const ptr_t newNnz)
{
    if(mat->rowind){
        mat->rowind = da_irealloc_a(mat->rowind, newNnz, "da_csr_matrixNNzRealloc: mat->rowind");
        if(mat->rowind == NULL)
            da_errexit( "da_csr_matrixNNzRealloc: Could not reallocate mat->rowind size %zu.\n", newNnz);
    }
    if(mat->rowval){
        mat->rowval = da_vrealloc_a(mat->rowval, newNnz, "da_csr_matrixNNzRealloc: mat->rowval");
        if(mat->rowval == NULL)
            da_errexit( "da_csr_matrixNNzRealloc: Could not reallocate mat->rowval size %zu.\n", newNnz);
    }
    if(mat->colind){
        mat->colind = da_irealloc_a(mat->colind, newNnz, "da_csr_matrixNNzRealloc: mat->colind");
        if(mat->colind == NULL)
            da_errexit( "da_csr_matrixNNzRealloc: Could not reallocate mat->colind size %zu.\n", newNnz);
    }
    if(mat->colval){
        mat->colval = da_vrealloc_a(mat->colval, newNnz, "da_csr_matrixNNzRealloc: mat->colval");
        if(mat->colval == NULL)
            da_errexit( "da_csr_matrixNNzRealloc: Could not reallocate mat->colval size %zu.\n", newNnz);
    }
}





/**************************************************************************/
/*! Reads a CSR matrix from the supplied file and stores it the matrix's
    forward structure.
    \param filename is the file that stores the data.
    \param format is either DA_FMT_METIS, DA_FMT_CLUTO,
           DA_FMT_CSR, DA_FMT_BINROW, DA_FMT_BINCOL
           specifying the type of the input format.
           The DA_FMT_CSR does not contain a header
           line, whereas the DA_FMT_BINROW is a binary format written
           by da_csr_Write() using the same format specifier.
    \param readvals is either 1 or 0, indicating if the CSR file contains
           values or it does not. It only applies when DA_FMT_CSR is
           used.
    \param numbering is either 1 or 0, indicating if the numbering of the
           indices start from 1 or 0, respectively. If they start from 1,
           they are automatically decremented during input so that they
           will start from 0. It only applies when DA_FMT_CSR is
           used.
    \returns the matrix that was read.
 */
/**************************************************************************/
da_csr_t* da_csr_Read(const char* const filename,
		const char format, char readvals, char numbering)
{
	ssize_t i, j, k, l, rid, len, nnz2, nr, nc, nz, rv;
	size_t nrows, ncols, nnz, nfields, fmt, ncon, lnlen, read, size;
	ptr_t *rowptr;
	idx_t *rowind;
	int32_t *iinds, *jinds, *bptr = NULL, *bind = NULL;
	val_t *rowval = NULL, *vals, fval;
	float *bval = NULL;
	char *line = NULL, *head, *tail, fmtstr[256];
	FILE *fpin;
	da_csr_t *mat = NULL;

	nrows = ncols = nnz = 0;

	if (!da_fexists(filename))
		da_errexit( "File %s does not exist!\n", filename);

	if (format == DA_FMT_IJV) {
		da_getfilestats(filename, &nrows, &nnz, NULL, NULL);

		if (readvals == 1 && 3 * nrows != nnz)
			da_errexit(
					"Error: The number of numbers (%zd %d) in the input file is not a multiple of 3.\n",
					nnz, readvals);
		if (readvals == 0 && 2 * nrows != nnz)
			da_errexit(
					"Error: The number of numbers (%zd %d) in the input file is not a multiple of 2.\n",
					nnz, readvals);

		mat = da_csr_Create();
		nnz = nrows;
		numbering = (numbering ? -1 : 0);
		iinds = da_imalloc(nnz, "iinds");
		jinds = da_imalloc(nnz, "jinds");
		vals  = (readvals ? da_fmalloc(nnz, "vals") : NULL);

		fpin = da_fopen(filename, "r", "da_csr_Read: fpin");
		for (nrows = 0, ncols = 0, i = 0; i < nnz; ++i) {
			if (readvals) {
				if (fscanf(fpin, "%d %d %f", &iinds[i], &jinds[i], &vals[i]) != 3)
					da_errexit( "Error: Failed to read (i, j, val) for nnz: %zd.\n", i);
			} else {
				if (fscanf(fpin, "%d %d", &iinds[i], &jinds[i]) != 2)
					da_errexit( "Error: Failed to read (i, j) value for nnz: %zd.\n", i);
			}
			iinds[i] += numbering;
			jinds[i] += numbering;

			if (nrows < iinds[i])
				nrows = iinds[i];
			if (ncols < jinds[i])
				ncols = jinds[i];
		}
		nrows++;
		ncols++;
		da_fclose(fpin);

	}

	if (format == DA_FMT_IJV) {
		/* convert (i, j, v) into a CSR matrix */
		mat->nrows = nrows;
		mat->ncols = ncols;
		rowptr = mat->rowptr = da_pnmalloc(mat->nrows + 1, "rowptr");
		rowind = mat->rowind = da_imalloc(nnz, "rowind");
		if (readvals)
			rowval = mat->rowval = da_vmalloc(nnz, "rowval");

		for (i = 0; i < nnz; ++i)
			rowptr[iinds[i]]++;
		CSRMAKE(i, mat->nrows, rowptr);

		for (i = 0; i < nnz; ++i) {
			rowind[rowptr[iinds[i]]] = jinds[i];
			if (readvals)
				rowval[rowptr[iinds[i]]] = vals[i];
			rowptr[iinds[i]]++;
		}
		CSRSHIFT(i, mat->nrows, rowptr);

		da_free((void **) &iinds, &jinds, &vals, LTERM);

		return mat;
	}

	if (format == DA_FMT_CLUTO) {
		fpin = da_fopen(filename, "r", "da_csr_Read: fpin");
		do {
			if (da_getline(&line, &lnlen, fpin) <= 0)
				da_errexit( "Premature end of input file: %s\n", filename);
		} while (line[0] == '%');

		if (sscanf(line, "%zu %zu %zu", &nrows, &ncols, &nnz) != 3)
			da_errexit( "Header line must contain 3 integers.\n");

		readvals  = 1;
		numbering = 1;
	}
	else if (format == DA_FMT_METIS) {
		fpin = da_fopen(filename, "r", "da_csr_Read: fpin");
		do {
			if (da_getline(&line, &lnlen, fpin) <= 0)
				da_errexit( "Premature end of input file: %s\n", filename);
		} while (line[0] == '%');

		fmt = ncon = 0;
		nfields = sscanf(line, "%zu %zu %zu %zu", &nrows, &nnz, &fmt, &ncon);
		if (nfields < 2)
			da_errexit( "Header line must contain at least 2 integers (#vtxs and #edges).\n");

		ncols = nrows;
		nnz *= 2;

		if (fmt > 111)
			da_errexit( "Cannot read this type of file format [fmt=%zu]!\n", fmt);

		sprintf(fmtstr, "%03zu", fmt%1000);
		readvals  = (fmtstr[2] == '1');
		numbering = 1;
		ncon      = (ncon == 0 ? 1 : ncon);
	}
	else {
		da_getfilestats(filename, &nrows, &nnz, NULL, NULL);

		if (readvals == 1 && nnz%2 == 1)
			da_errexit( "Error: The number of numbers (%zd %d) in the input file is not even.\n", nnz, readvals);
		if (readvals == 1)
			nnz = nnz/2;
		fpin = da_fopen(filename, "r", "da_csr_Read: fpin");
	}

	mat = da_csr_Create();

	mat->nrows = nrows;
	mat->ncols = ncols;

	rowptr = mat->rowptr = da_pmalloc(nrows+1, "da_csr_Read: rowptr");
	rowind = mat->rowind = da_imalloc(nnz, "da_csr_Read: rowind");
	if (readvals != 2)
		rowval = mat->rowval = da_vsmalloc(nnz, 1.0, "da_csr_Read: rowval");

	/*----------------------------------------------------------------------
	 * Read the sparse matrix file
	 *---------------------------------------------------------------------*/
	numbering = (numbering ? - 1 : 0);
	for (ncols=0, rowptr[0]=0, k=0, i=0; i<nrows; ++i) {
		do {
			if (da_getline(&line, &lnlen, fpin) == -1)
			    goto endrows;  // allow for CLUTO matrix with more rows than its last row
//				da_errexit( "Premature end of input file while reading row %d\n", i);
		} while (line[0] == '%');

		head = line;
		tail = NULL;

		/* Read the rest of the row */
		while (1) {
			len = (int)strtol(head, &tail, 0);
			if (tail == head)
				break;
			head = tail;

			if ((rowind[k] = len + numbering) < 0)
				da_errexit( "Error: Invalid column number %d at row %zd.\n", len, i);

			ncols = da_max(rowind[k], ncols);

			if (readvals == 1) {
#ifdef __MSC__
				fval = (float)strtod(head, &tail);
#else
				fval = strtof(head, &tail);
#endif
				if (tail == head)
					da_errexit( "Value could not be found for column! Row:%zd, NNZ:%zd\n", i, k);
				head = tail;

				rowval[k] = fval;
			}
			k++;
		}
		rowptr[i+1] = k;
	}

	endrows:

	for(;i < nrows; ++i)
	    rowptr[i+1] = k;

	if (format == DA_FMT_METIS) {
		ASSERT(ncols+1 == mat->nrows);
		mat->ncols = mat->nrows;
	}
	else {
		mat->ncols = da_max(mat->ncols, ncols+1);
	}

	if (k != nnz)
		da_errexit( "da_csr_Read: Something wrong with the number of nonzeros in "
				"the input file. NNZ=%zd, ActualNNZ=%zd.\n", nnz, k);

	da_fclose(fpin);

	da_free((void **)&line, LTERM);

	return mat;
}



/**************************************************************************/
/*! Writes the row-based structure of a matrix into a file.
    \param mat is the matrix to be written,
    \param filename is the name of the output file.
    \param format is one of: DA_FMT_CLUTO, DA_FMT_CSR,
           DA_FMT_BINROW, DA_FMT_BINCOL.
    \param writevals is either 1 or 0 indicating if the values will be
           written or not. This is only applicable when DA_FMT_CSR
           is used.
    \param numbering is either 1 or 0 indicating if the internal 0-based
           numbering will be shifted by one or not during output. This
           is only applicable when DA_FMT_CSR is used.
 */
/**************************************************************************/
void da_csr_Write(const da_csr_t* const mat, const char* const filename,
		const char format, char writevals, char numbering)
{
	ssize_t i, j;
	size_t nnz;
	idx_t len;
	idx_t *ind;
	ptr_t *ptr;
	val_t *val;
	int32_t nr, nc, vId;
	int32_t edge[2];
	da_csr_t *tmp = NULL;
	FILE *fpout = NULL;

	if (!mat->rowval)
		writevals = 0;

	nr  = mat->nrows;
	nc  = mat->ncols;
	nnz = mat->rowptr[mat->nrows];
	ptr = mat->rowptr;
	ind = mat->rowind;
	val = mat->rowval;



	assert(mat->nrows <= INT32_MAX);
	assert(mat->ncols <= INT32_MAX);
	assert(mat->rowptr[mat->nrows] <= SIZE_MAX);

	if (format == DA_FMT_METIS)
		da_errexit( "METIS output format is not supported.\n");

	if (filename)
		fpout = da_fopen(filename, "w", "da_csr_Write: fpout");
	else
		fpout = stdout;

	if (format == DA_FMT_IJV) {
		numbering = (numbering ? 1 : 0);
		for (i = 0; i < mat->nrows; ++i) {
			for (j = ptr[i]; j < ptr[i+1]; ++j) {
				if (writevals)
					fprintf(fpout, "%zd\t%d\t%.19g\n", i + numbering,
							ind[j] + numbering, val[j]);
				else
					fprintf(fpout, "%zd\t%d\n", i + numbering,
							ind[j] + numbering);
			}
		}

		if (fpout)
			da_fclose(fpout);
		return;
	}

	if (format == DA_FMT_CLUTO) {
		fprintf(fpout, "%d %d %zu\n",
				nr, nc, nnz);
		if (mat->rowval)
			writevals = 1;
		numbering = 1;
	}

	for (i = 0; i < nr; ++i) {
		for (j = ptr[i]; j < ptr[i+1]; ++j) {
			fprintf(fpout, " %d",
					ind[j] + (numbering ? 1 : 0));
			if (writevals)
				fprintf(fpout, " %f", val[j]);
		}
		fprintf(fpout, "\n");
	}

    if (fpout){
        da_fclose(fpout);
    }
}


/**************************************************************************/
/*! Prints the row based representation of the matrix to screen.
    \param mat is the matrix to be printed.
*/
/**************************************************************************/
void da_csr_PrintInfo(
        const da_csr_t* const mat,
        const char* const name,
        const char* const suffix)
{
    if(!mat){
        printf("%s%s[NO MATRIX]%s",
            name != NULL ? name : "",
            name == NULL ? "" : name[strlen(name)-1] == ' ' ? "" : " ",
            suffix ? suffix : "");
    } else {
        printf("%s%s[" PRNT_IDXTYPE ", " PRNT_IDXTYPE ", " PRNT_PTRTYPE "]",
                name != NULL ? name : "",
                name == NULL ? "" : name[strlen(name)-1] == ' ' ? "" : " ",
                mat->nrows, mat->ncols,
                mat->rowptr ? mat->rowptr[mat->nrows] :
                mat->colptr ? mat->colptr[mat->ncols] : 0);
        if(!mat->rowptr && ! mat->colptr)
            printf(" [!!MISSING STRUCTURE!!]");
        if(suffix)
            printf("%s", suffix);
    }

    fflush(stdout);
}



/**************************************************************************/
/*! Prints the row based representation of the matrix to screen.
    \param mat is the matrix to be printed.
*/
/**************************************************************************/
void da_csr_Print(const da_csr_t* const mat)
{
	da_csr_Write(mat, NULL, DA_FMT_CLUTO, 1, 1);
}


/*************************************************************************/
/*! Check if a text (non-binary) text file containing a csr is in CLUTO or CSR format.
    \param file is the matrix file to be checked.
    \return the CSR format: MV_FMT_CLUTO or MV_FMT_CSR
 */
/*************************************************************************/
char da_csr_isClutoOrCsr(const char* const file)
{
	size_t nrows, nnz;

	da_getfilestats(file, &nrows, &nnz, NULL, NULL);
	return (nnz%2 == 1) ? DA_FMT_CLUTO : DA_FMT_CSR;
}




/*************************************************************************/
/*! Compacts the column-space of the matrix by removing empty columns.
    As a result of the compaction, the column numbers are renumbered.
    The compaction operation is done in place and only affects the row-based
    representation of the matrix.
    The new columns are ordered in decreasing frequency.

    \param mat the matrix whose empty columns will be removed.
 */
/**************************************************************************/
void da_csr_CompactColumns(da_csr_t* const mat)
{
	ssize_t i;
	idx_t nrows, ncols, nncols;
	ptr_t *rowptr;
	idx_t *rowind, *colmap;
	da_iikv_t *clens;

	nrows  = mat->nrows;
	ncols  = mat->ncols;
	rowptr = mat->rowptr;
	rowind = mat->rowind;

	colmap = da_imalloc(ncols, "da_csr_CompactColumns: colmap");

	clens = da_iikvmalloc(ncols, "da_csr_CompactColumns: clens");
	for (i=0; i<ncols; ++i) {
		clens[i].key = i;
		clens[i].val = 0;
	}

	for (i=0; i<rowptr[nrows]; ++i)
		clens[rowind[i]].val++;
	da_iikvsortd(ncols, clens);

	for (nncols=0, i=0; i<ncols; ++i) {
		if (clens[i].val > 0)
			colmap[clens[i].key] = nncols++;
		else
			break;
	}

	for (i=0; i<rowptr[nrows]; ++i)
		rowind[i] = colmap[rowind[i]];

	mat->ncols = nncols;

	da_free((void **)&colmap, &clens, LTERM);
}

/*************************************************************************/
/*! Compacts the row-space of the matrix by removing empty rows.
    As a result of the compaction, the row numbers are renumbered.
    The compaction operation is done in place and only affects the row-based
    representation of the matrix.

    \param mat the matrix whose empty rows will be removed.
 */
/**************************************************************************/
void da_csr_CompactRows(da_csr_t* const mat)
{
    ssize_t i, j;
    idx_t nrows;
    ptr_t *rowptr;

    nrows  = mat->nrows;
    rowptr = mat->rowptr;

    for(j=0, i=0; i < nrows; i++){
        rowptr[j] = rowptr[i];
        if(rowptr[i+1] > rowptr[i])
            j++;
    }

    rowptr[j+1] = rowptr[nrows];
    mat->nrows = j;
    mat->rowptr = da_prealloc(mat->rowptr, j+1, "da_csr_CompactRows: mat->rowptr realloc");
}


/*************************************************************************/
/*! Sorts the indices in increasing order (whether matrix has values or not)
    \param mat the matrix itself,
    \param what is either DA_ROW or DA_COL indicating which set of
           indices to sort.
*/
/**************************************************************************/
void da_csr_SortIndices(da_csr_t* const mat, const char what)
{
	ptr_t n, nn=0;
	ptr_t *ptr;
	idx_t *ind;
	val_t *val;

	switch (what) {
	case DA_ROW:
		if (!mat->rowptr)
			da_errexit( "Row-based view of the matrix does not exists.\n");

		n   = mat->nrows;
		ptr = mat->rowptr;
		ind = mat->rowind;
		val = mat->rowval;
		break;

	case DA_COL:
		if (!mat->colptr)
			da_errexit( "Column-based view of the matrix does not exists.\n");

		n   = mat->ncols;
		ptr = mat->colptr;
		ind = mat->colind;
		val = mat->colval;
		break;

	default:
		da_errexit( "Invalid index type of %d.\n", what);
		return;
	}

    ssize_t i, j, k;
    da_pikv_t *cand;
    val_t *tval = NULL;

    for (i=0; i<n; ++i)
        nn = da_max(nn, ptr[i+1]-ptr[i]);

    cand = da_pikvmalloc(nn, "da_csr_SortIndices: cand");
    if(val){
        tval = da_vmalloc(nn, "da_csr_SortIndices: tval");

        for (i=0; i<n; ++i) {
            for (k=0, j=ptr[i]; j<ptr[i+1]; ++j) {
                if (j > ptr[i] && ind[j] < ind[j-1]){
                    k = 1; /* an inversion */
                }
                cand[j-ptr[i]].key = j-ptr[i];
                cand[j-ptr[i]].val = ind[j];
                tval[j-ptr[i]]     = val[j];
            }
            if (k) {
                da_pikvsorti(ptr[i+1]-ptr[i], cand);
                for (j=ptr[i]; j<ptr[i+1]; ++j) {
                    ind[j] = cand[j-ptr[i]].val;
                    val[j] = tval[cand[j-ptr[i]].key];
                }
            }
        }

    } else {

        for (i=0; i<n; ++i) {
            for (k=0, j=ptr[i]; j<ptr[i+1]; ++j) {
                if (j > ptr[i] && ind[j] < ind[j-1]){
                    k = 1; /* an inversion */
                }
                cand[j-ptr[i]].key = j-ptr[i];
                cand[j-ptr[i]].val = ind[j];
            }
            if (k) {
                da_pikvsorti(ptr[i+1]-ptr[i], cand);
                for (j=ptr[i]; j<ptr[i+1]; ++j){
                    ind[j] = cand[j-ptr[i]].val;
                }
            }
        }

    }
    da_free((void **)&cand, &tval, LTERM);


}



/*************************************************************************/
/*! Checks that an index is sorted
    \param mat the matrix itself,
    \param what is either DA_ROW or DA_COL indicating which set of
           indices to check.
*/
/**************************************************************************/
char da_csr_CheckSortedIndex(da_csr_t* const mat, const char what)
{
	ptr_t n, nn=0;
	ptr_t *ptr;
	idx_t *ind;
	ssize_t i, j;
	char k=1;

	switch (what) {
	case DA_ROW:
		if (!mat->rowptr)
			da_errexit( "Row-based view of the matrix does not exists.\n");

		n   = mat->nrows;
		ptr = mat->rowptr;
		ind = mat->rowind;
		break;

	case DA_COL:
		if (!mat->colptr)
			da_errexit( "Column-based view of the matrix does not exists.\n");

		n   = mat->ncols;
		ptr = mat->colptr;
		ind = mat->colind;
		break;

	default:
		da_errexit( "Invalid index type of %d.\n", what);
		return k;
	}

	for (k=1, i=0; i < n && k == 1; ++i) {
		for (j=ptr[i]; j < ptr[i+1]; ++j) {
			if (j > ptr[i] && ind[j] < ind[j-1]){
				k = 0; /* an inversion */
				break;
			}
		}
	}

	return k;
}



/*************************************************************************/
/*! Sorts the values (and associated indices) in increasing or decreasing order
    \param mat the matrix itself,
    \param what is either DA_ROW or DA_COL indicating which set of
           values/indices to sort.
    \param mrl maximum row length (set to 0 to find it from the data)
    \param how Sort increasing (DA_SORT_I) or decreasing (DA_SORT_D)
*/
/**************************************************************************/
void da_csr_SortValues(
        da_csr_t * const mat,
        char const what,
        idx_t const mrl,
        char const how)
{
    ptr_t ii, n, nn;
    ptr_t *ptr;
    idx_t *ind;
    val_t *val;

    switch (what) {
    case DA_ROW:
        if (!mat->rowptr)
            da_errexit( "Row-based view of the matrix does not exists.\n");

        n   = mat->nrows;
        ptr = mat->rowptr;
        ind = mat->rowind;
        val = mat->rowval;
        break;

    case DA_COL:
        if (!mat->colptr)
            da_errexit( "Column-based view of the matrix does not exists.\n");

        n   = mat->ncols;
        ptr = mat->colptr;
        ind = mat->colind;
        val = mat->colval;
        break;

    default:
        da_errexit( "Invalid index type of %d.\n", what);
        return;
    }

    if(!val)
        da_errexit("da_csr_SortValues: values not present in %s structure of the matrix.",
                what == DA_ROW ? "row" : "col ");


    if(mrl > 0){
        nn = mrl;
    } else {
        for (nn=0, ii=0; ii<n; ++ii){
            if(ptr[ii+1]-ptr[ii] > nn){
                nn = ptr[ii+1]-ptr[ii];
            }
        }
    }

    ssize_t i, j, k, s, e;
    da_ivkv_t *cand;

    cand = da_ivkvmalloc(nn, "da_csr_SortValues: cand");

    if(how == DA_SORT_I){
        for (i=0; i<n; ++i) {
            s = ptr[i];
            e = ptr[i+1];
            if(e-s < 2){
                continue;
            }
            cand[0].key = ind[s];
            for (k=0, j=s+1; j < e; ++j) {
                if (val[j] < val[j-1]){
                    k = 1; /* an inversion */
                    break;
                }
                cand[j-s].key = ind[j];
            }
            if (k) {
                for ( ; j < e; ++j) {
                    cand[j-s].key = ind[j];
                }
                for (j=s; j < e; ++j) {
                    cand[j-s].val = val[j];
                }
                da_ivkvsorti(e-s, cand);
                for (j=s; j < e; ++j) {
                    ind[j] = cand[j-s].key;
                }
                for (j=s; j < e; ++j) {
                    val[j] = cand[j-s].val;
                }
            }
        }
    } else {
        for (i=0; i<n; ++i) {
            s = ptr[i];
            e = ptr[i+1];
            if(e-s < 2){
                continue;
            }
            cand[0].key = ind[s];
            for (k=0, j=s+1; j < e; ++j) {
                if (val[j] > val[j-1]){
                    k = 1; /* an inversion */
                    break;
                }
                cand[j-s].key = ind[j];
            }
            if (k) {
                for ( ; j < e; ++j) {
                    cand[j-s].key = ind[j];
                }
                for (j=s; j < e; ++j) {
                    cand[j-s].val = val[j];
                }
                da_ivkvsortd(e-s, cand);
                for (j=s; j < e; ++j) {
                    ind[j] = cand[j-s].key;
                }
                for (j=s; j < e; ++j) {
                    val[j] = cand[j-s].val;
                }
            }
        }
    }
    da_free((void **)&cand, LTERM);


}


/*************************************************************************/
/*! Creates a row/column index from the column/row data.
    \param mat the matrix itself,
    \param what is either DA_ROW or DA_COL indicating which index
           will be created.
 */
/**************************************************************************/
void da_csr_CreateIndex(da_csr_t* const mat, const char what)
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
		rptr = mat->colptr = da_pnmalloc(nr+1, "da_csr_CreateIndex: rptr");
		rind = mat->colind = da_imalloc(fptr[nf], "da_csr_CreateIndex: rind");
		rval = mat->colval = (fval ? da_vmalloc(fptr[nf], "da_csr_CreateIndex: rval") : NULL);
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
		rptr = mat->rowptr = da_pnmalloc(nr+1, "da_csr_CreateIndex: rptr");
		rind = mat->rowind = da_imalloc(fptr[nf], "da_csr_CreateIndex: rind");
		rval = mat->rowval = (fval ? da_vmalloc(fptr[nf], "da_csr_CreateIndex: rval") : NULL);
		break;
	default:
		da_errexit( "Invalid index type of %d.\n", what);
		return;
	}


	for (i=0; i<nf; ++i) {
		for (j=fptr[i]; j<fptr[i+1]; ++j)
			rptr[find[j]]++;
	}
	CSRMAKE(i, nr, rptr);

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



/*************************************************************************/
/*! Creates a row/column index from the column/row data.
    \param mat the matrix itself,
    \param what is either DA_ROW or DA_COL indicating which index
           will be created.
 */
/**************************************************************************/
void da_csr_CreateIndex_a(da_csr_t* const mat, const char what)
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
        rptr = mat->colptr = da_pnmalloc_a(nr+1, "da_csr_CreateIndex: rptr");
        rind = mat->colind = da_imalloc_a(fptr[nf], "da_csr_CreateIndex: rind");
        rval = mat->colval = (fval ? da_vmalloc_a(fptr[nf], "da_csr_CreateIndex: rval") : NULL);
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
        rptr = mat->rowptr = da_pnmalloc_a(nr+1, "da_csr_CreateIndex: rptr");
        rind = mat->rowind = da_imalloc_a(fptr[nf], "da_csr_CreateIndex: rind");
        rval = mat->rowval = (fval ? da_vmalloc_a(fptr[nf], "da_csr_CreateIndex: rval") : NULL);
        break;
    default:
        da_errexit( "Invalid index type of %d.\n", what);
        return;
    }


    for (i=0; i<nf; ++i) {
        for (j=fptr[i]; j<fptr[i+1]; ++j)
            rptr[find[j]]++;
    }
    CSRMAKE(i, nr, rptr);

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
            for (j=fptr[i]; j<fptr[i+1]; ++j){
                k = find[j];
                rind[rptr[k]++] = i;

            }
        }
    }

    CSRSHIFT(i, nr, rptr);

}


/*************************************************************************/
/*! Normalizes the rows/columns of the matrix to be unit
    length.
    \param mat the matrix itself,
    \param what indicates what will be normalized and is obtained by
           specifying DA_ROW, DA_COL, DA_ROW|DA_COL.
    \param norm indicates what norm is to normalize to, 1: 1-norm, 2: 2-norm
 */
/**************************************************************************/
void da_csr_Normalize(da_csr_t* const mat, const char what, const char norm)
{
	ssize_t i, j;
	idx_t n;
	ptr_t *ptr;
	val_t *val;
	double sum;

	if ((what & DA_ROW) && mat->rowval) {
		n   = mat->nrows;
		ptr = mat->rowptr;
		val = mat->rowval;

        for (i=0; i<n; ++i) {
            for (sum=0.0, j=ptr[i]; j<ptr[i+1]; ++j){
                if (norm == 2)
                    sum += val[j]*val[j];
                else if (norm == 1)
                    sum += val[j]; /* assume val[j] > 0 */
            }
            if (sum > 0) {
                if (norm == 2)
                    sum=1.0/sqrt(sum);
                else if (norm == 1)
                    sum=1.0/sum;
                for (j=ptr[i]; j<ptr[i+1]; ++j)
                    val[j] *= sum;

            }
        }
	}

	if ((what & DA_COL) && mat->colval) {
		n   = mat->ncols;
		ptr = mat->colptr;
		val = mat->colval;

		for (i=0; i<n; ++i) {
			for (sum=0.0, j=ptr[i]; j<ptr[i+1]; ++j)
				if (norm == 2)
					sum += val[j]*val[j];
				else if (norm == 1)
					sum += val[j];
			if (sum > 0) {
				if (norm == 2)
					sum=1.0/sqrt(sum);
				else if (norm == 1)
					sum=1.0/sum;
				for (j=ptr[i]; j<ptr[i+1]; ++j)
					val[j] *= sum;
			}
		}
	}

}


/*************************************************************************/
/*! Scale matrix by IDF
    \param mat the matrix itself,
 */
/**************************************************************************/
void da_csr_Scale(da_csr_t* const mat)
{
	ssize_t i, j;
	idx_t nrows, ncols, nnzcols, bgfreq;
	ptr_t *rowptr;
	idx_t *rowind, *collen;
	val_t *rowval;
	double *cscale, maxtf;

	nrows  = mat->nrows;
    ncols  = mat->ncols;
	rowptr = mat->rowptr;
	rowind = mat->rowind;
	rowval = mat->rowval;

    cscale = da_dmalloc(ncols, "da_csr_Scale: cscale");
    collen = da_inmalloc(ncols, "da_csr_Scale: collen");

    for (i=0; i<nrows; ++i) {
        for (j=rowptr[i]; j<rowptr[i+1]; ++j)
            collen[rowind[j]]++;
    }

    for (i=0; i<ncols; ++i)
        cscale[i] = (collen[i] > 0 ? log(1.0*nrows/collen[i]) : 0.0);

    for (i=0; i<nrows; ++i) {
        for (j=rowptr[i]; j<rowptr[i+1]; ++j)
            rowval[j] *= cscale[rowind[j]];
    }

    da_free((void **)&cscale, &collen, LTERM);

}



/*************************************************************************/
/*! Computes the cosine similarity between two rows/columns

    \param mat the matrix itself. The routine assumes that the indices
           are sorted in increasing order.
    \param r1 is the first row/column,
    \param r2 is the second row/column,
    \param what is either DA_ROW or DA_COL indicating the type of
           objects between the similarity will be computed,
    \returns the similarity between the two rows/columns.
 */
/**************************************************************************/
val_t da_csr_ComputeSimilarity(const da_csr_t* const mat,
		const idx_t rc1, const idx_t rc2, const char what)
{
	idx_t nind1, nind2, i1, i2;
	idx_t *ind1, *ind2;
	val_t *val1, *val2, stat1, stat2, sim;

	switch (what) {
	case DA_ROW:
		if (!mat->rowptr)
			da_errexit( "Row-based view of the matrix does not exists.\n");
		nind1 = mat->rowptr[rc1+1]-mat->rowptr[rc1];
		nind2 = mat->rowptr[rc2+1]-mat->rowptr[rc2];
		ind1  = mat->rowind + mat->rowptr[rc1];
		ind2  = mat->rowind + mat->rowptr[rc2];
		val1  = mat->rowval + mat->rowptr[rc1];
		val2  = mat->rowval + mat->rowptr[rc2];
		break;

	case DA_COL:
		if (!mat->colptr)
			da_errexit( "Column-based view of the matrix does not exists.\n");
		nind1 = mat->colptr[rc1+1]-mat->colptr[rc1];
		nind2 = mat->colptr[rc2+1]-mat->colptr[rc2];
		ind1  = mat->colind + mat->colptr[rc1];
		ind2  = mat->colind + mat->colptr[rc2];
		val1  = mat->colval + mat->colptr[rc1];
		val2  = mat->colval + mat->colptr[rc2];
		break;

	default:
		da_errexit( "Invalid index type of %d.\n", what);
		return 0.0;
	}


    sim = stat1 = stat2 = 0.0;
    i1 = i2 = 0;
    while (i1<nind1 && i2<nind2) {
        if (i1 == nind1) {
            stat2 += val2[i2]*val2[i2];
            i2++;
        }
        else if (i2 == nind2) {
            stat1 += val1[i1]*val1[i1];
            i1++;
        }
        else if (ind1[i1] < ind2[i2]) {
            stat1 += val1[i1]*val1[i1];
            i1++;
        }
        else if (ind1[i1] > ind2[i2]) {
            stat2 += val2[i2]*val2[i2];
            i2++;
        }
        else {
            sim   += val1[i1]*val2[i2];
            stat1 += val1[i1]*val1[i1];
            stat2 += val2[i2]*val2[i2];
            i1++;
            i2++;
        }
    }
    sim = (stat1*stat2 > 0.0 ? sim/sqrt(stat1*stat2) : 0.0);

	return sim;

}






static char da_iarreq(const size_t n, const idx_t* const x, const idx_t* const y)
{
  size_t i;

  for (i=0; i<n; i++)\
    if(x[i] != y[i]) return 0;

  return 1;
}


static char da_parreq(const size_t n, const ptr_t* const x, const ptr_t* const y)
{
  size_t i;

  for (i=0; i<n; i++)\
    if(x[i] != y[i]) return 0;

  return 1;
}

static char da_varreq_p(const size_t n, const val_t* const x, const val_t* const y, const double p)\
{
  size_t i;

  for (i=0; i<n; i++)
    if(((x[i] - y[i]) >= 0 ? (x[i] - y[i]) : -(x[i] - y[i])) > p) return 0;

  return 1;
}


/*************************************************************************/
/*! Check of two matrices are equal. Note that one of the matrices may have an
 * 	additional index created to facilitate the comparison
    \param a is the matrix to be compared.
    \param b is the matrix to be compared against.
    \param p is the precision to be used in the comparison.
    \returns 1 if matrices have the same elements, 0 otherwise.
 */
/**************************************************************************/
char da_csr_Compare(const da_csr_t* const a, const da_csr_t* const b, const double p)
{
	if(!(a && b)) return 0;
	if(a->ncols != b->ncols || a->nrows != b->nrows) return 0;
	ptr_t nnz = 0;

	assert((a->rowptr && b->rowptr) || (a->colptr && b->colptr));

	if(a->rowptr && b->rowptr){
		if(a->rowptr[a->nrows] != b->rowptr[b->nrows])
			return 0;
		nnz = a->rowptr[a->nrows];
		if(!da_parreq(a->nrows+1, a->rowptr, b->rowptr))
			return 0;
		if(!da_iarreq(nnz, a->rowind, b->rowind))
			return 0;
		if(a->rowval && b->rowval && !da_varreq_p(nnz, a->rowval, b->rowval, p))
			return 0;
		else if((a->rowval && !b->rowval) || (!a->rowval && b->rowval))
			return 0;
	} else if(a->colptr && b->colptr){
		if(a->colptr[a->ncols] != b->colptr[b->ncols])
			return 0;
		nnz = a->colptr[a->ncols];
		if(!da_parreq(a->ncols+1, a->colptr, b->colptr))
			return 0;
		if(!da_iarreq(nnz, a->colind, b->colind))
			return 0;
		if(a->rowval && b->rowval && !da_varreq_p(nnz, a->colval, b->colval, p))
			return 0;
		else if((a->rowval && !b->rowval) || (!a->rowval && b->rowval))
			return 0;
	} else {
		return 0;
	}
	return 1;
}



/**
 * Increase the space needed for storing nnzs in a csr matrix
 * 	\param mat The matrix to have its nnz space re-allocated
 * 	\param newNnz The new size of the row/col ind/val arrays
 */
void da_csr_Grow(da_csr_t* const mat, const ptr_t newNnz)
{
	if(mat->rowind){
		mat->rowind = da_irealloc(mat->rowind, newNnz, "da_csr_matrixNNzRealloc: mat->rowind");
		if(mat->rowind == NULL)
			da_errexit( "da_csr_matrixNNzRealloc: Could not reallocate mat->rowind size %zu.\n", newNnz);
	}
	if(mat->rowval){
		mat->rowval = da_vrealloc(mat->rowval, newNnz, "da_csr_matrixNNzRealloc: mat->rowval");
		if(mat->rowval == NULL)
			da_errexit( "da_csr_matrixNNzRealloc: Could not reallocate mat->rowval size %zu.\n", newNnz);
	}
	if(mat->colind){
		mat->colind = da_irealloc(mat->colind, newNnz, "da_csr_matrixNNzRealloc: mat->colind");
		if(mat->colind == NULL)
			da_errexit( "da_csr_matrixNNzRealloc: Could not reallocate mat->colind size %zu.\n", newNnz);
	}
	if(mat->colval){
		mat->colval = da_vrealloc(mat->colval, newNnz, "da_csr_matrixNNzRealloc: mat->colval");
		if(mat->colval == NULL)
			da_errexit( "da_csr_matrixNNzRealloc: Could not reallocate mat->colval size %zu.\n", newNnz);
	}
}


void da_csr_Transpose(
        da_csr_t * const mat)
{
    idx_t n;
    ptr_t *cptr;
    idx_t *cind, *cids;
    val_t *cval, *cnorms, *csizes, *csums, *cvols, *cwgts;

    if(!mat->colptr){
        da_csr_CreateIndex(mat, DA_COL);
    }

    cptr   = mat->colptr;
    cind   = mat->colind;
    cval   = mat->colval;
    cnorms = mat->cnorms;

    n          = mat->nrows;
    mat->nrows = mat->ncols;
    mat->ncols = n;

    mat->colptr = mat->rowptr;
    mat->colind = mat->rowind;
    mat->colval = mat->rowval;
    mat->cnorms = mat->rnorms;

    mat->rowptr = cptr;
    mat->rowind = cind;
    mat->rowval = cval;
    mat->rnorms = cnorms;
}
