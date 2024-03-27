export DATA_PATH="/home/oop/dev/data"
export CKPT_PATH="/home/oop/dev/data/test/ckpt"
export LOGS_PATH="/home/oop/dev/data/test/logs"
docker build -t jax-mnist -f Dockerfile.test .
docker run \
    -it \
    --rm \
    -p 5555:5555 \
    --gpus 0 \
    -v ${DATA_PATH}:/data \
    -v ${CKPT_PATH}:/ckpt \
    -v ${LOGS_PATH}:/logs \
    -e RUN_NAME=mnisttest \
    jax-mnist