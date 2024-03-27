# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax
FROM nvcr.io/nvidia/jax:23.08-py3
ENV RUN_NAME="test"
RUN pip install --upgrade pip
RUN pip install \
    "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    jaxlib \
    flax
RUN mkdir /data
RUN mkdir /ckpt
RUN mkdir /logs
RUN mkdir /src
WORKDIR /src
COPY jamba_mnist.py /src/jamba_mnist.py
CMD ["python", "jamba_mnist.py"]