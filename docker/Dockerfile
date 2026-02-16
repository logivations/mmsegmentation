# It is important that cuda supports the video card architectures that are important to us:
# NVIDIA GeForce RTX 5090               - sm_120 (Blackwell) - requires CUDA 12.8+
# NVIDIA GeForce RTX 3060 / RTX 3060 Ti - sm_86
# NVIDIA GeForce RTX 2080 Ti            - sm_75
# NVIDIA A100-SXM4-40GB                 - sm_80
# Build with a version tag
# docker build -t quay.io/logivations/ml_all:LS_mmseg_cuda128 -f docker/Dockerfile .

ARG PYTORCH="2.7.1"
ARG CUDA="12.8"
ARG CUDNN="9"


FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;12.0+PTX"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-dev  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# Install MMCV
RUN pip install openmim
RUN mim install mmengine "mmcv>=2.0.0,<2.2.0"

COPY . /mmsegmentation
WORKDIR /mmsegmentation


# Install MMSegmentation
ENV FORCE_CUDA="1"
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -e .
RUN pip install git+https://github.com/facebookresearch/segment-anything.git
RUN pip install git+https://github.com/ChaoningZhang/MobileSAM.git