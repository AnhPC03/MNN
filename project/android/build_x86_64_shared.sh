#!/bin/bash
cmake ../../../ \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="x86_64" \
-DANDROID_STL=c++_shared \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_NATIVE_API_LEVEL=android-21  \
-DANDROID_TOOLCHAIN=clang \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3 \
-DOpenCV_DIR=/home/anhpc03/opencv_build/opencv/build

make -j4
