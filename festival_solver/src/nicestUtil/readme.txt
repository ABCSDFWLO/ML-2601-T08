23jan26

This directory holds my favorite configuration

So in the minimal "util0.cpp" I just return 1 
in "get_number_of_cores" for 
osx/linux rather than getting detailed 
#processors + #GbMem.

Copy these to src if you want to get all details
on all 3 OS's. memory.cpp does this cleanly.

Note that OSX segfaults when trying more than 1 core,
but I did complete the detection code
for number of cores anyway.

Addendum

Note that the preferred compiler-directive choices seem to be

_WIN32 || _WIN64			for Windows (1st: either; 2nd: 64 only)

__linux__					for linux

__APPLE__  &&  __MACH__	for OSX


