chmod +x cuda_11.4.3_470.82.01_linux.run
./cuda_11.4.3_470.82.01_linux.run
echo 'export PATH=/usr/local/cuda-11.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
ldconfig

tar -xzvf cudnn-11.4-linux-x64-v8.2.4.15.tgz
cp -P cuda/include/cudnn.h /usr/local/cuda-11.4/include
cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.4/lib64/
chmod a+r /usr/local/cuda-11.4/lib64/libcudnn*

nvidia-smi
nvcc -V
