docker run -it --gpus all \
  --shm-size=8g \
  --rm \
  --name "seg_train" \
  -v "/data:/data/" \
  -v "/data/mmsegmentation:/mmsegmentation/" \
  -w /mmsegmentation \
  quay.io/logivations/ml_all:LS_mmseg_latest