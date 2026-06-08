# Using Mikhael Goikhman's -Wno
# option to allow using g++
# rather than gcc


HOME=/home/rufas
export PATH=$HOME/opt/gnat-x86_64-linux-14.2.0-1/bin:$PATH


g++  -o festival_gnu \
-I. \
-fpermissive -m64 -O3 \
-DLINK -Wno-error=changes-meaning \
-DTHREADS *.cpp \
-lpthread -lm

mv festival_gnu ../bin/

