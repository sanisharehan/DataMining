/*!
\file  io.c
\brief Various file I/O functions.

This file contains various functions that perform I/O.

Ported from George Karypis' GKlib library by David C. Anastasiu, with permission, in Aug 2013.

\date Started 4/10/95
\author George
\version\verbatim $Id: io.c 15225 2013-09-25 14:49:08Z karypis $ \endverbatim
 */

#ifdef HAVE_GETLINE
/* Get getline to be defined. */
#define _GNU_SOURCE
#include <stdio.h>
#undef _GNU_SOURCE
#endif

#include "includes.h"



/*************************************************************************
 * This function checks if a file exists
 **************************************************************************/
int da_fexists(const char* const fname)
{
    struct stat status;

    if (stat(fname, &status) == -1)
        return 0;

    return S_ISREG(status.st_mode);
}


/*************************************************************************
 * This function checks if a directory exists
 **************************************************************************/
int da_dexists(const char* const dirname)
{
    struct stat status;

    if (stat(dirname, &status) == -1)
        return 0;

    return S_ISDIR(status.st_mode);
}


/*************************************************************************/
/*! \brief Returns the size of the file in bytes

This function returns the size of a file as a 64 bit integer. If there
were any errors in stat'ing the file, -1 is returned.
\note That due to the -1 return code, the maximum file size is limited to
      63 bits (which I guess is okay for now).
 */
/**************************************************************************/
ssize_t da_getfsize(const char* const filename)
{
    struct stat status;

    if (stat(filename, &status) == -1)
        return -1;

    return (size_t)(status.st_size);
}



/*************************************************************************
 * This function takes in a potentially full path specification of a file
 * and just returns a string containing just the basename of the file.
 * The basename is derived from the actual filename by stripping the last
 * .ext part.
 **************************************************************************/
char* da_getbasename(const char* const path)
{
    char *startptr = NULL, *endptr;
    char *basename;

    if ((startptr = strrchr(path, '/')) == NULL)
        *startptr = *path;
    else
        startptr = startptr+1;

    basename = da_strdup(startptr);

    if ((endptr = strrchr(basename, '.')) != NULL)
        *endptr = '\0';

    return basename;
}

/*************************************************************************
 * This function takes in a potentially full path specification of a file
 * and just returns a string corresponding to its file extension. The
 * extension of a file is considered to be the string right after the
 * last '.' character.
 **************************************************************************/
char* da_getextname(const char* const path)
{
    char *startptr;

    if ((startptr = strrchr(path, '.')) == NULL)
        return da_strdup(path);
    else
        return da_strdup(startptr+1);
}

/*************************************************************************
 * This function takes in a potentially full path specification of a file
 * and just returns a string containing just the filename.
 **************************************************************************/
char* da_getfilename(const char* const path)
{
    char *startptr;

    if ((startptr = strrchr(path, '/')) == NULL)
        return da_strdup(path);
    else
        return da_strdup(startptr+1);
}

/*************************************************************************
 * This function takes in a potentially full path specification of a file
 * and extracts the directory path component if it exists, otherwise it
 * returns "./" as the path. The memory for it is dynamically allocated.
 **************************************************************************/
char* da_getpathname(const char* const path)
{
    char *endptr, *tmp;

    if (path == NULL || (endptr = strrchr(path, '/')) == NULL) {
        return da_strdup(".");
    }
    else  {
        tmp = da_strdup(path);
        *(strrchr(tmp, '/')) = '\0';
        return tmp;
    }
}



/*************************************************************************
 * This function creates a path
 **************************************************************************/
int da_mkpath(const char* const path)
{
    char tmp[2048];

    sprintf(tmp, "mkdir -p %s", path);
    return system(tmp);
}


/*************************************************************************
 * This function deletes a directory tree and all of its contents
 **************************************************************************/
int da_rmpath(const char* const path)
{
    char tmp[2048];

    sprintf(tmp, "rm -r %s", path);
    return system(tmp);
}




/*************************************************************************/
/*! This function gets some basic statistics about the file. Same as GK's version
 *  but handles comment lines.
    \param fname is the name of the file
    \param r_nlines is the number of lines in the file. If it is NULL,
           this information is not returned.
    \param r_ntokens is the number of tokens in the file. If it is NULL,
           this information is not returned.
    \param r_max_nlntokens is the maximum number of tokens in any line
           in the file. If it is NULL this information is not returned.
    \param r_nbytes is the number of bytes in the file. If it is NULL,
           this information is not returned.
 */
/*************************************************************************/
void da_getfilestats(const char* const fname, size_t* r_nlines, size_t* r_ntokens,
        size_t* r_max_nlntokens, size_t* r_nbytes)
{
    size_t nlines=0, ntokens=0, max_nlntokens=0, nbytes=0, oldntokens=0, nread;
    int32_t intoken=0, lineType=0; //lineType 0-not started, 1-started ok line, 2-comment line
    char buffer[2049], *cptr;
    FILE *fpin;

    fpin = da_fopen(fname, "r", "da_getfilestats");

    while (!feof(fpin)) {
        nread = fread(buffer, sizeof(char), 2048, fpin);
        nbytes += nread;

        buffer[nread] = '\0';  /* There is space for this one */
        for (cptr=buffer; *cptr!='\0'; ++cptr) {
            if (*cptr == '%' && lineType == 0){
                lineType = 2;
            }
            else if (*cptr == '\n') {
                if(lineType != 2){
                    nlines++;
                    ntokens += intoken;
                }
                intoken = 0;
                lineType = 0;
                if (max_nlntokens < ntokens-oldntokens)
                    max_nlntokens = ntokens-oldntokens;
                oldntokens = ntokens;
            }
            else if (*cptr == ' ' || *cptr == '\t') {
                ntokens += intoken;
                intoken = 0;
            }
            else if(lineType != 2){
                intoken = 1;
                lineType = 1;
            }
        }
    }
    ntokens += intoken;
    if (max_nlntokens < ntokens-oldntokens)
        max_nlntokens = ntokens-oldntokens;

    da_fclose(fpin);

    if (r_nlines != NULL)
        *r_nlines  = nlines;
    if (r_ntokens != NULL)
        *r_ntokens = ntokens;
    if (r_max_nlntokens != NULL)
        *r_max_nlntokens = max_nlntokens;
    if (r_nbytes != NULL)
        *r_nbytes  = nbytes;
}




/*************************************************************************
 * This function opens a file
 **************************************************************************/
FILE* da_fopen(const char* const fname, const char* const mode, const char* const msg)
{
    FILE *fp;
    char errmsg[8192];

    fp = fopen(fname, mode);
    if (fp != NULL)
        return fp;

    sprintf(errmsg,"file: %s, mode: %s, [%s]", fname, mode, msg);
    perror(errmsg);
    da_errexit("Failed on da_fopen()\n");

    return NULL;
}


/*************************************************************************
 * This function closes a file
 **************************************************************************/
void da_fclose(FILE* fp)
{
    fclose(fp);
}


/*************************************************************************/
/*! This function is a wrapper around the read() function that ensures
    that all data is been read, by issueing multiple read requests.
 */
/*************************************************************************/
ssize_t da_read(const int fd, void* vbuf, const size_t count)
{
    char *buf = (char *)vbuf;
    ssize_t rsize, tsize=count;

    do {
        if ((rsize = read(fd, buf, tsize)) == -1)
            return -1;
        buf   += rsize;
        tsize -= rsize;
    } while (tsize > 0);

    return count;
}



/*************************************************************************/
/*! This function is the GKlib implementation of glibc's getline()
    function.
    \returns -1 if the EOF has been reached, otherwise it returns the
             number of bytes read.
 */
/*************************************************************************/
ssize_t da_getline(char** lineptr, size_t* n, FILE* stream)
{
#ifdef HAVE_GETLINE
    return getline(lineptr, n, stream);
#else
    size_t i;
    int ch;

    if (feof(stream))
        return -1;

    /* Initial memory allocation if *lineptr is NULL */
    if (*lineptr == NULL || *n == 0) {
        *n = 1024;
        *lineptr = da_malloc((*n)*sizeof(char), "da_getline: lineptr");
    }

    /* get into the main loop */
    i = 0;
    while ((ch = getc(stream)) != EOF) {
        (*lineptr)[i++] = (char)ch;

        /* reallocate memory if reached at the end of the buffer. The +1 is for '\0' */
        if (i+1 == *n) {
            *n = 2*(*n);
            *lineptr = da_realloc(*lineptr, (*n)*sizeof(char), "da_getline: lineptr");
        }

        if (ch == '\n')
            break;
    }
    (*lineptr)[i] = '\0';

    return (i == 0 ? -1 : i);
#endif
}


/*************************************************************************/
/*! This function reads the contents of a text file and returns it in the
    form of an array of strings.
    \param fname is the name of the file
    \param r_nlines is the number of lines in the file. If it is NULL,
           this information is not returned.
 */
/*************************************************************************/
char** da_readfile(const char* const fname, size_t* r_nlines)
{
    size_t lnlen=0, nlines=0;
    char *line=NULL, **lines=NULL;
    FILE *fpin;

    da_getfilestats(fname, &nlines, NULL, NULL, NULL);
    if (nlines > 0) {
        lines = (char **)da_malloc(nlines*sizeof(char *), "da_readfile: lines");

        fpin = da_fopen(fname, "r", "da_readfile");
        nlines = 0;
        while (da_getline(&line, &lnlen, fpin) != -1) {
            da_strtprune(line, "\n\r");
            lines[nlines++] = da_strdup(line);
        }
        da_fclose(fpin);
    }

    da_free((void **)&line, LTERM);

    if (r_nlines != NULL)
        *r_nlines  = nlines;

    return lines;
}

