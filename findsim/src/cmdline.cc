/*!
 \file  cmdline.c
 \brief This file contains functions for parsing command-line arguments

 \author David C. Anastasiu
 */
#include "includes.h"


/*-------------------------------------------------------------------
 * Command-line options 
 *-------------------------------------------------------------------*/
static struct da_option long_options[] = {
    {"mode",              1,      0,      CMD_MODE},
    {"m",                 1,      0,      CMD_MODE},
    {"k",                 1,      0,      CMD_K},
    {"e",                 1,      0,      CMD_EPSILON},
    {"eps",               1,      0,      CMD_EPSILON},
    {"epsion",            1,      0,      CMD_EPSILON},
    {"verb",              1,      0,      CMD_VERBOSITY},
    {"version",           0,      0,      CMD_VERSION},
	{"v",                 1,      0,      CMD_VERIFY},
	{"stats",             0,      0,      CMD_STATS},
    {"fldelta",           1,      0,      CMD_FLDELTA},
    {"fd",                1,      0,      CMD_FLDELTA},
    {"fmtRead",           1,      0,      CMD_FMT_READ},
	{"readZidx",          0,      0,      CMD_FMT_READ_NUM},
	{"readVals",          1,      0,      CMD_READ_VALS},
	{"fmtWrite",          1,      0,      CMD_FMT_WRITE},
	{"writeZidx",         0,      0,      CMD_FMT_WRITE_NUM},
	{"writeVals",         1,      0,      CMD_WRITE_VALS},
	{"help",              0,      0,      CMD_HELP},
	{"h",                 0,      0,      CMD_HELP},
	{0,                   0,      0,      0}
};


/*-------------------------------------------------------------------
 * Mini help
 *-------------------------------------------------------------------*/
static char helpstr[][100] =
{
PROGRAM_NAME " - Compute the exact Cosine K-Nearest Neighbor graph",
"                with a minimum similarity eps for a set of sparse vectors.",
" ",
"Usage: " PROGRAM_NAME " [options] mode input-file [output-file]",
" ",
" <input/output-file> should be in CSR, CLUTO, or IJV (Coordinate) format.",
" Input is assumed to be a document term-frequency matrix. Term frequencies will be scaled",
" by IDF before computing similarities.",
" If no <output-file> is specified, the output will not be saved. K-NNG output will be ",
" sparse vectors, sorted in decreasing similarity order.",
" ",
" Options",
" ",
"  -mode:",
"    ij       Build graph using IdxJoin (full sparse dot-products). Default.",
" ",
"  (utility modes):",
"    info     Get information about the sparse matrix in input-file (output-file ignored).",
"    testeq   Test whether matrix in input-file is the same as that in output-file.",
"             Differences will be printed out to stdout.",
"    io       Transform sparse matrix in input file and write to output-file in",
"             specified format.",
"    recall   Compute recall of a knng solution given true values. ",
"             Usage: findsim recall <true_results> <test_results> ",
" ",
"  -k=int",
"     Number of neighbors to return for each row in the Min-eps K-Nearest Neighbor Graph.",
"     Default value is 10.",
" ",
"  -eps=float",
"     Minimum similarity for neighbors.",
"     Default value is 0.5. Must be non-negative.",
" ",
"  -v=string",
"     Verification file containing a true Min-eps K-Nearest Neighbor Graph. Must be in CSR format.",
"     Default value is NULL (no verification).",
" ",
"  -fmtRead=string",
"     What format is the dataset stored in: clu, csr, ijv.",
"     See README for format definitions.",
"     Default value is 0 (detect from extension).",
" ",
"  -readZidx",
"     Column ids start with 0 instead of 1.",
" ",
"  -readVals=int",
"     Read values from file.",
"     Default value is 1.",
" ",
"  -fmtWrite=string",
"     What format should the output file be written in. See -fmtRead for values.",
"     Default value is ijv.",
" ",
"  -writeZidx",
"     Column ids start with 0 instead of 1.",
" ",
"  -writeVals=int",
"     Write values to file.",
"     Default value is 1.",
" ",
"  -stats",
"     Display additional statistics for the matrix (applies to mode 'info' only).",
" ",
"  -fldelta=int",
"     Float delta used when testing equality of real numbers. (testeq mode only)",
"     Default value is 1e-4.",
" ",
"  -verb=int",
"     Specifies the level of debugging information to be displayed:",
"         0 = NONE, 1 = INFO",
"     Default value is 0 (NONE).",
" ",
"  -version",
"     Prints version information.",
" ",
"  -help, -h",
"     Prints this message.",
""
};


const da_StringMap_t mode_options[] = {
  {"ij",                MODE_IDXJOIN},
  {"fast",              MODE_FAST},
  {"dynamic",           MODE_DYNAMIC},
  {"idxjoin",           MODE_IDXJOIN},
  /** Add new modes here if desired. Mode constants are defined in defs.h */

  {"recall",            MODE_RECALL},
  {"eq",                MODE_TESTEQUAL},
  {"testeq",            MODE_TESTEQUAL},
  {"io",                MODE_IO},
  {"info",              MODE_INFO},
  {NULL,                 0}
};

const da_StringMap_t fmt_options[] = {
  {"clu",               DA_FMT_CLUTO},
  {"csr",               DA_FMT_CSR},
  {"met",               DA_FMT_METIS},
  {"ijv",               DA_FMT_IJV},
  {NULL,                 0}
};



/*************************************************************************/
/*! This is the entry point of the command-line argument parser          */
/*************************************************************************/
void cmdline_parse(params_t *params, int argc, char *argv[])
{
	idx_t i;
	long l;
	int32_t c, option_index;


	/* initialize the params data structure */
	memset(params, 0, sizeof(params_t));

	params->verbosity    = 1;
	params->stats        = 0;

	params->mode         = MODE_IDXJOIN;
    params->k            = 10;
	params->epsilon      = 0.5;
    
    params->fldelta      = 1e-4;
	
    params->iFile        = NULL;
	params->fmtRead      = 0;
	params->readVals     = 1;
	params->readNum      = 1;
	params->oFile        = NULL;
	params->fmtWrite     = 0;
	params->writeVals    = 1;
	params->writeNum     = 1;
    params->vFile        = NULL;

	params->filename     = da_cmalloc(1024, "cmdline_parse: filename");
    params->docs         = NULL;
	params->neighbors    = NULL;

	/* timers */
	params->timer_1      = 0.0;
	params->timer_2      = 0.0;
	params->timer_3      = 0.0;
	params->timer_4      = 0.0;
	params->timer_5      = 0.0;
	params->timer_6      = 0.0;
	params->timer_7      = 0.0;
	params->timer_8      = 0.0;
	params->timer_9      = 0.0;
	params->timer_global = 0.0;

	/* Parse the command line arguments  */
	while ((c = da_getopt_long_only(argc, argv, "", long_options, &option_index)) != -1) {
		switch (c) {

        case CMD_MODE:
            params->mode = da_getStringID(mode_options, da_optarg);
            if (params->mode == -1)
                da_errexit("Invalid mode %s.\n", argv[da_optind]);
            break;

        case CMD_K:
            if (da_optarg) {
                if ((params->k = atoi(da_optarg)) < 1)
                    da_errexit("Invalid -k. Must be greater than 0.\n");
            }
            break;

        case CMD_EPSILON:
            if (da_optarg) {
                if ((params->epsilon = atof(da_optarg)) < 0 || params->epsilon > 1)
                    da_errexit("The -alpha value must be in [0,1].\n");
            }
            break;


		case CMD_VERBOSITY:
			if (da_optarg) {
				if ((params->verbosity = atoi(da_optarg)) < 0)
					da_errexit("The -verbosity value must be non-negative.\n");
			}
			break;

		case CMD_FMT_READ:
			if (da_optarg) {
				if ((params->fmtRead = da_getStringID(fmt_options, da_optarg)) == -1)
					da_errexit("Invalid -fmtRead. Options are: clu, csr, met, binr, and binc.\n");
			}
			break;

		case CMD_READ_VALS:
			if (da_optarg) {
				if ((params->readVals = atoi(da_optarg)) < 0 || params->readVals > 1)
					da_errexit("Invalid -readVals. Must be 0 or 1.\n");
			}
			break;

		case CMD_FMT_READ_NUM:
			params->readNum = 0;
			break;


		case CMD_FMT_WRITE:
			if (da_optarg) {
				if ((params->fmtWrite = da_getStringID(fmt_options, da_optarg)) == -1)
					da_errexit("Invalid -fmtWrite. Options are: clu, csr, met, binr, and binc.\n");
			}
			break;

		case CMD_WRITE_VALS:
			if (da_optarg) {
				if ((params->writeVals = atoi(da_optarg)) < 0 || params->writeVals > 1)
					da_errexit("Invalid -writeVals. Must be 0 or 1.\n");
			}
			break;

		case CMD_FMT_WRITE_NUM:
			params->writeNum = 0;
			break;
    
        case CMD_STATS:
            params->stats = 1;
            break;

        case CMD_FLDELTA:
            if (da_optarg) {
                if ((params->fldelta = atof(da_optarg)) <= 0)
                    da_errexit("The -fldelta value must be greater than 0.\n");
            }
            break;

		case CMD_VERSION:
			printf("%s (%d.%d.%d), vInfo: [%s]\n", argv[0], VER_MAJOR, VER_MINOR,
						VER_SUBMINOR, VER_COMMENT);
			exit(EXIT_SUCCESS);
			break;

        case CMD_VERIFY:
            params->vFile = da_strdup(da_optarg);
            if(!da_fexists(params->vFile))
                da_errexit("The -v parameter requires a valid verification file. %s is not a file.\n", params->vFile);
            break;

		case CMD_HELP:
			for (i=0; strlen(helpstr[i]) > 0; i++)
				printf("%s\n", helpstr[i]);
			exit(EXIT_SUCCESS);
			break;

		default:
			printf("Illegal command-line option(s)\nUse %s -help for a summary of the options.\n", argv[0]);
			exit(EXIT_FAILURE);
		}
	}

	/* Get the operation to be performed */
	if(argc > da_optind)
		params->iFile     = da_strdup(argv[da_optind++]);

	if(params->iFile && !da_fexists(params->iFile))
		da_errexit("Invalid input file %s!\n", params->iFile);
    params->fmtRead = da_getFileFormat(params->iFile, params->fmtRead);

	if(argc > da_optind){		/* output file passed in */
        params->oFile     = da_strdup(argv[da_optind++]);
        if(strcmp(params->iFile, params->oFile) == 0)
            da_errexit("The input file and the output file cannot be the same.");

		if(params->mode != MODE_INFO && params->mode != MODE_TESTEQUAL && params->mode != MODE_RECALL) {
            params->fmtWrite = da_getFileFormat(params->oFile, params->fmtWrite);
        }
	}

	if(!params->oFile && params->mode == MODE_TESTEQUAL)
        da_errexit("Output file required for mode %s!\n", da_getStringKey(mode_options, params->mode));


	/* print the command line */
	if(params->verbosity > 0){
		for (i=0; i<argc; i++)
			printf("%s ", argv[i]);
		printf("\n");
	}

}


