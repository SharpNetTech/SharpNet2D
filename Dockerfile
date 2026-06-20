FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS build

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV TCNN_CUDA_ARCHITECTURES="52,60,61,70,80"

RUN pip install --no-cache-dir --verbose --root-user-action=ignore \
    "igraph==0.11.9" \
    "matplotlib==3.10.6" \
    "networkx==3.5" \
    "open3d==0.19.0" \
    "pyhocon==0.3.59" \
    "svgpathtools==1.7.1" \
    "tensorboard==2.20.0" \
    "tqdm==4.67.1" \
    "trimesh==4.8.1" \
    "git+https://github.com/NVlabs/tiny-cuda-nn.git@v2.0#subdirectory=bindings/torch"

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS release
COPY --from=build /opt/conda /opt/conda
