# this is a "free" way to build on OSX (without Xcode)

#!/bin/sh

# use this to ensure a complete recompilation:
if [ -d ./obj/ ]; then
	rm ./obj/*
else
	mkdir obj
fi


export GROOT=$HOME/opt/GNAT/2018/bin
export PATH=$GROOT:$PATH
export SDKROOT=$(xcrun --show-sdk-path)

export FWROOT=/System/Library/Frameworks


# pow in libm

$GROOT/g++  -o festival_osx \
-I.  \
-fpermissive -m64 -O3 \
-W -Wall -Wextra -pedantic -Wno-unused-variable \
-Wno-unused-parameter \
-DTHREADS *.cpp \
-L$SDKROOT/usr/lib \
-lm -lz \
-pthread

mv festival_osx ../bin/


