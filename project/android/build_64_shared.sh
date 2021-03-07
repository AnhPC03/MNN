#!/bin/bash
cmake ../../../ \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_STL=c++_shared \
-DMNN_USE_LOGCAT=false \
-DMNN_BUILD_BENCHMARK=ON \
-DANDROID_NATIVE_API_LEVEL=android-21  \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3 \
-DOpenCV_DIR=/home/anhpc03/opencv_build/opencv/build

make -j4
