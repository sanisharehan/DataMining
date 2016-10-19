

#include "includes.h"




/*-------------------------------------------------------------
 * Timing inline functions
 *-------------------------------------------------------------*/
/**
 * Returns time stored in wall timer in seconds
 */
inline double timer_get(
        double tmr)
{
    if( tmr < MAX_RUNTIME ){  /** stop timer if started */
        tmr += da_wallclock();
    }
    return tmr * 0.000001;
}

/**
 * Returns time stored in cpu timer in seconds
 */
inline double cputimer_get(
        double tmr)
{
    if( tmr < MAX_RUNTIME ){  /** stop timer if started */
        tmr += da_cpuclock();
    }
    return tmr * 0.000001;
}

/*************************************************************************
* This function returns the CPU seconds
**************************************************************************/
double da_wallclock(void)
{
#ifdef __GNUC__
    struct timeval ctime;

    gettimeofday(&ctime, NULL);

    return (1.0e+6 * (double)ctime.tv_sec + (double)ctime.tv_usec);
#else
    return (double)time(NULL);
#endif
}


/*************************************************************************
 * This function returns the CPU seconds
 **************************************************************************/
double da_cpuclock(void)
{
#if defined(WIN32) || defined(__MINGW32__)
    return((double) clock()/CLOCKS_PER_SEC);
#else
    struct rusage r;

    getrusage(RUSAGE_SELF, &r);
    return (1.0e+6 * (r.ru_utime.tv_sec + r.ru_stime.tv_sec) + (r.ru_utime.tv_usec + r.ru_stime.tv_usec));
#endif
}


/*************************************************************************/
/*! Prints a timer in human readable format. Timers should be started and
 * stopped with da_startwctimer(t) and da_stopwctimer(t) respectively.
 *
 *  \param name is the timer name
 *  \param time is the time in seconds
 */
/*************************************************************************/
void da_printTime(
        char const * const name,
        double time)
{
    time = timer_get(time);
    if(time < 60){
        printf("%s %.2fs\n", name, time);
    } else if(time < 3600){
        int32_t min = time/60;
        time -= min*60;
        printf("%s %dm %.2fs\n", name, min, time);
    } else {
        int32_t hours = time/3600;
        time -= hours*3600;
        int32_t min = time/60;
        time -= min*60;
        printf("%s %dh %dm %.2fs\n", name, hours, min, time);
    }
}
void da_printTimer(
        char const * const name,
        double time)
{
    da_printTime( name, timer_get(time) );
}

/************************************************************************/
/*! \brief Prints a timer in long format, including human readable and
 * computer readable time. Timers should be started and stopped with
 * da_startwctimer(t) and da_stopwctimer(t) respectively.

    \param name is the Timer name, a string to be printed before the timer.
    \param time is the time that should be reported.
*/
/*************************************************************************/
void da_printTimeLong(
        char const * const name,
        double time)
{
    double t = time;
    if(time < 60){
        printf("%s %.4f (%.2fs)\n", name, t, time);
    } else if(time < 3600){
        int32_t min = time/60;
        time -= min*60;
        printf("%s %.4f (%dm %.2fs)\n", name, t, min, time);
    } else {
        int32_t hours = time/3600;
        time -= hours*3600;
        int32_t min = time/60;
        time -= min*60;
        printf("%s %.4f (%dh %dm %.2fs)\n", name, t, hours, min, time);
    }
}
void da_printTimerLong(
        char const * const name,
        double time)
{
    da_printTimeLong( name, timer_get(time) );
}


