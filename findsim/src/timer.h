

#ifndef _FINDSIM_TIMER_
#define _FINDSIM_TIMER_

#include "includes.h"


/*-------------------------------------------------------------
 * Timing functions
 *-------------------------------------------------------------*/
double da_wallclock(void);
double da_cpuclock(void);
void da_printTime(
        char const * const name,
        double time);
void da_printTimer(
        char const * const name,
        double time);
void da_printTimeLong(
        char const * const name,
        double time);
void da_printTimerLong(
        char const * const name,
        double time);
double timer_get(
        double tmr);
double cputimer_get(
        double tmr);


/*-------------------------------------------------------------
 * Timing defines
 *-------------------------------------------------------------*/
#define MAX_RUNTIME  -10*365*24*60*60 /** Timing will fail if execution takes longer than 10 years */



/*-------------------------------------------------------------
 * Timing macros
 *-------------------------------------------------------------*/
#define cputimer_clear(tmr) (tmr = 0.0)
#define cputimer_start(tmr) do {if(tmr > MAX_RUNTIME){tmr -= da_cpuclock();}} while(0)
#define cputimer_stop(tmr)  do {if(tmr < MAX_RUNTIME){tmr += da_cpuclock();}} while(0)

#define timer_clear(tmr) (tmr = 0.0)
#define timer_start(tmr) do {if(tmr > MAX_RUNTIME){tmr -= da_wallclock();}} while(0)
#define timer_stop(tmr)  do {if(tmr < MAX_RUNTIME){tmr += da_wallclock();}} while(0)

#ifdef EXTRATIMES
    #define etimer_clear timer_clear
    #define etimer_start timer_start
    #define etimer_stop timer_stop
    #define etimer_get timer_get
    #define ecputimer_clear cputimer_clear
    #define ecputimer_start cputimer_start
    #define ecputimer_stop cputimer_stop
    #define ecputimer_get cputimer_get
    #define da_printETimer da_printTimer
    #define da_printETimerLong da_printTimerLong
#else
    #define etimer_clear(tmr) {}
    #define etimer_start(tmr) {}
    #define etimer_stop(tmr) {}
    #define etimer_get(tmr) (0.0)
    #define ecputimer_clear(tmr) {}
    #define ecputimer_start(tmr) {}
    #define ecputimer_stop(tmr) {}
    #define ecputimer_get(tmr) (0.0)
    #define da_printETimer {}
    #define da_printETimerLong {}
#endif

#endif
