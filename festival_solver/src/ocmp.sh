# this is a "free" way to build on OSX (without Xcode)


export GROOT=$HOME/opt/GNAT/2018/bin
export PATH=$GROOT:$PATH


$GROOT/g++  -o festival_osx \
-I.  \
-fpermissive -m64 -O3 \
-DTHREADS *.cpp \
-pthread

mv festival_osx ../bin/

