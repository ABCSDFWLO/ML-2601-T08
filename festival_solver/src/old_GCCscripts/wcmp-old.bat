
rem First, run setpath.bat [for mingw64 OR try mine]

rem Yaron uses mingw64

rem gcc  g++
gcc -o festival ^
-I. -I\msys64\mingw64\include ^
-fpermissive -m64 -O3 ^
-DTHREADS *.cpp

move festival.exe ..\bin\

