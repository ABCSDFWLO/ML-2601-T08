
rem First, run setpath.bat [for mingw64 OR try mine]

g++ -o festival ^
-I. -I\msys64\mingw64\include ^
-fpermissive -m64 -O3 ^
-DLINK -Wno-error=changes-meaning ^
-DTHREADS *.cpp

move festival.exe ..\bin\

