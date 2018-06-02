nvcc = /usr/local/cuda-8.0/bin/nvcc
cudalib = /usr/local/cuda-8.0/lib64
tensorflow = /usr/local/lib/python2.7/dist-packages/tensorflow/include
#TF_LIB=/usr/local/lib/python2.7/dist-packages/tensorflow/core
#TF_INC=/usr/local/lib/python2.7/dist-packages/tensorflow/include
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
i=$1
o=${i/.cc/.so}
all: depthestimate/tf_nndistance_so.so depthestimate/render_balls_so.so
.PHONY : all

depthestimate/tf_nndistance_so.so: depthestimate/tf_nndistance_g.cu.o depthestimate/tf_nndistance.cpp
	g++ -std=c++11 depthestimate/tf_nndistance.cpp depthestimate/tf_nndistance_g.cu.o -o depthestimate/tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -I $TF_INC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -L $TF_LIB  -O2 -D_GLIBCXX_USE_CXX11_ABI=0  -L /usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework

depthestimate/tf_nndistance_g.cu.o: depthestimate/tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o depthestimate/tf_nndistance_g.cu.o depthestimate/tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -L$TF_LIB -ltensorflow_framework 

depthestimate/render_balls_so.so: depthestimate/render_balls_so.cpp
	g++ -std=c++11 depthestimate/render_balls_so.cpp -o depthestimate/render_balls_so.so -shared -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB  -O2 -D_GLIBCXX_USE_CXX11_ABI=0  -L /usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework



