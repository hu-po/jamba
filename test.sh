export DATA_PATH="/home/oop/dev/data"
docker build -t jax-mnist -f Dockerfile.test .
docker run \
    -it \
    --rm \
    -p 5555:5555 \
    --gpus 0 \
    -v ${DATA_PATH}:/data \
    -e RUN_NAME=mnisttest \
    jax-mnist