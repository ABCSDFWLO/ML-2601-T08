Permalink:

https://sourceforge.net/projects/festival3os/files/latest/download


# Festival3OS
	Mods to Festival for OSX, Linux, & Windows
	from fastrgv


**ver 1.0.2 -- 7feb2026

* Cleaned up the code somewhat.


**ver 1.0.1 -- 5feb2026

* Added generic memory-check module that works on OSX, Windows, & linux.
* Generalized compiler directives for 3 or more systems.
* Added pthread-attribute setting necessary for proper multi-threading under OSX.


**ver 1.0.0 -- 23jan2026

* First release with minimal changes to Festival.

### More Details

Festival is the well known, highly regarded sokoban solver written by Yaron Shoham. It is the first program that solves all 90 levels of the XSokoban benchmark.

link:

https://festival-solver.site/

### Changes

I downloaded the most recent version available on 22jan2026 and tried to make minimal changes, mostly in the compiler-directives, that allow building on 3 operating systems.

I did what works for me on my test systems. Of course the format of the directives had to be revised since they originally assumed either Windows xor Linux systems.

I show how to build on 3 systems using free software that is non-proprietary.

I have created many apps that run on OSX, Linux, & Windows, most of which are complex OpenGL graphics games, and puzzles, including 3 sokoban platforms and one sokoban solver written in Ada. I also created multi-threaded OpenAL sound utilities for the 3 OS's using C++ and Ada.

Thus, I call your attention to a highly portable module "memory.hpp" that has proven its reliability in determining total system memory and available memory on all 3 systems. It is now used in the festival3os code.

Finally, it seems OSX has a ridiculously small default pthread-stack-size which had to be adjusted larger in order to avoid a segfault. And for that value, I used the default linux pthread-stack-size. (I struggled with this bit of esoterica for quite a while.)


### Some Basic Commandline Parms

* -time f (sec)
* -cores i
* -level i

I added 3 example run scripts for each system that demonstrate requesting 1 or 2 cores, as well as a "0" version that gives Festival the option to choose how many cores to use.

For windows, these are:

* w1runlev.bat
* w2runlev.bat
* w0runlev.bat

These each take 2 parameters:

* puzzleFileName (string)
* level (int)

EG:
	w0runlev.bat puzzles\xsok_90.sok 26

will solve level 26 from XSokoban using Festival-defaults.


### Contact

Open source developers are welcome to help improve or extend this app.
Send comments, suggestions or questions to:

* fastrgv@gmail.com


