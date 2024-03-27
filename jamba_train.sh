export DATA_PATH="/home/oop/dev/data/mnist"
export CKPT_PATH="/home/oop/dev/data/jamba/ckpt"
export LOGS_PATH="/home/oop/dev/data/jamba/logs"
docker build -t jamba -f Dockerfile .
docker run \
    -it \
    --rm \
    -p 5555:5555 \
    --gpus 0 \
    -v ${DATA_PATH}:/data \
    -v ${CKPT_PATH}:/ckpt \
    -v ${LOGS_PATH}:/logs \
    -e RUN_NAME=test \
    jamba