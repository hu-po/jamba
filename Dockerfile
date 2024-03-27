FROM nvcr.io/nvidia/jax:23.08-py3
# FROM ghcr.io/nvidia/jax:base
COPY mnist_classifier.py /app/mnist_classifier.py
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install jaxlib
RUN pip install flax
WORKDIR /app
CMD ["python", "mnist_classifier.py"]
