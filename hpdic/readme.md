# HPDIC MOD of FAISS
We assume you are using (e.g., Chameleon Cloud `gigaio05`) Ubuntu 24.04, NVIDIA A100 GPU (40 GB RAM, Driver 560.35.05, CUDA 12.6), 256 GB RAM, Intel(R) Xeon(R) Platinum 8380 CPU (160 Cores).

## Recompile C++
```bash
cd ~/faiss
make -C build -j $(nproc) faiss
```

## Reinstall Python package
```bash
cd build/faiss/python
python setup.py install
```

## If you change header files, do the following, you need to do both C++ and Python steps.

## Installation Steps
```bash
python3 script/test_gpu.py
python3 -m venv myenv
source myenv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
sudo apt install -y cmake swig g++ libopenblas-dev libmkl-dev git
cd ~
git clone https://github.com/hpdic/faiss.git
cd faiss
source ~/myenv/bin/activate
which python 
# 输出应该是 /home/cc/myenv/bin/python
rm -rf build
cmake -B build . \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DFAISS_ENABLE_RAFT=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="80" \
    -DPython_EXECUTABLE=$(which python)
make -C build -j $(nproc)
cd build/faiss/python
python setup.py install
cd build/faiss/python
python setup.py install
```