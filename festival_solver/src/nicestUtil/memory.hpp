/*
	10jan2024
	Modified by fastrgv to repair OSX code.
	Eliminated aix code.
	License: either CCA-3.0 or GPLv3
*/

/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#include <stdio.h>
#include <unistd.h>

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif



/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
size_t getPeakRSS( )
{
#if defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage( RUSAGE_SELF, &rusage );
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}





/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
size_t getCurrentRSS( )
{
#if defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE* fp = NULL;
    if ( (fp = fopen( "/proc/self/statm", "r" )) == NULL )
        return (size_t)0L;      /* Can't open? */
    if ( fscanf( fp, "%*s%ld", &rss ) != 1 )
    {
        fclose( fp );
        return (size_t)0L;      /* Can't read? */
    }
    fclose( fp );

// that ugly fscanf-format mess above simply grabs the second integer in the file !!

/* debug stuff:
#include <iostream>
using std::cout;
using std::endl;
size_t trss, tpag;
trss=(size_t)rss;
tpag=(size_t)sysconf( _SC_PAGESIZE);
cout<<"trss="<<trss<<endl;
cout<<"tpag="<<tpag<<endl;
*/

	// I claim this underestimates used memory; thusly it overestimates available memory.
    return (size_t)rss * (size_t)sysconf( _SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}



// Addendum of fastrgv begin #################################################

#if defined(_WIN32)

	unsigned long long getTotalSystemMemoryMb()
	{
		 MEMORYSTATUSEX status;
		 status.dwLength = sizeof(status);
		 GlobalMemoryStatusEx(&status);
		 return status.ullTotalPhys / 1e6;
	}


	unsigned long long getAvailSystemMemoryMb()
	{
		 MEMORYSTATUSEX status;
		 status.dwLength = sizeof(status);
		 GlobalMemoryStatusEx(&status);
		 return status.ullAvailPhys / 1e6;
	}


#elif defined(__APPLE__) && defined(__MACH__)

	unsigned long long getTotalSystemMemoryMb()
	{
		 long pages = sysconf(_SC_PHYS_PAGES);
		 long page_size = sysconf(_SC_PAGE_SIZE);
		 return pages * page_size / 1e6;
	}


#include <mach/vm_statistics.h>
#include <mach/mach_types.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>

	unsigned long long getAvailSystemMemoryMb() {

		long long free_memory = 0;

		vm_size_t page_size;
		mach_port_t mach_port;
		mach_msg_type_number_t count;
		vm_statistics64_data_t vm_stats;

		mach_port = mach_host_self();
		count = sizeof(vm_stats) / sizeof(natural_t);
		if (KERN_SUCCESS == host_page_size(mach_port, &page_size) &&
			 KERN_SUCCESS == host_statistics64(mach_port, HOST_VM_INFO,
								  (host_info64_t)&vm_stats, &count))
		{
			// Ok:
			//free_memory = (int64_t)vm_stats.free_count * (int64_t)page_size;

			// better:
			free_memory = 
				((int64_t)vm_stats.free_count + (int64_t)vm_stats.inactive_count)
				* (int64_t)page_size;

		}

		return free_memory / 1e6;

	}



#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

	unsigned long long getTotalSystemMemoryMb()
	{
		 long pages = sysconf(_SC_PHYS_PAGES);
		 long page_size = sysconf(_SC_PAGE_SIZE);
		 return pages * page_size / 1e6;
	}


	unsigned long long getAvailSystemMemoryMb()
	{
		unsigned long long available;
		unsigned long long sysmemMb = getTotalSystemMemoryMb();
		size_t currentSize = getCurrentRSS();

		available = sysmemMb - currentSize/1e3;
		return available;
	}


#endif

// Addendum of fastrgv end ###################################################



