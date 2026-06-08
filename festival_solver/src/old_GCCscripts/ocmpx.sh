# this is a "free" way to build on OSX (without Xcode)

# use this to ensure a complete recompilation:
if [ -d ./obj/ ]; then
	rm ./obj/*
else
	mkdir obj
fi


export GROOT=$HOME/opt/GNAT/2018/bin
export PATH=$GROOT:$PATH
export SDKROOT=$(xcrun --show-sdk-path)



$GROOT/g++  -o festival_osx \
-std=c++11 \
-I. \
-fpermissive \
-DTHREADS *.cpp \
-lm -lz \
-pthread

mv festival_osx ../bin/


