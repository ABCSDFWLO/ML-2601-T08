# my original script using gcc

# use this to ensure a complete recompilation:
if [ -d ./obj/ ]; then
	rm ./obj/*
else
	mkdir obj
fi



gcc  -o festival_gnu \
-I. -I/usr/include \
-fpermissive -m64 -O3 \
-DTHREADS *.cpp \
-lpthread -lm

mv festival_gnu ../bin/

