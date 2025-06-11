#!/bin/bash

NVINFER_FILE=""
ONNX_FILENAME=""
CLASSES=()
RES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nvinfer-file)
            NVINFER_FILE="$2"
            shift 2
            ;;
        --onnx-filename)
            ONNX_FILENAME="$2"
            shift 2
            ;;
        --classes)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                CLASSES+=("$1")
                shift
            done
            ;;
        --res)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                RES+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

CLASSES=$(IFS=';' ; echo "${CLASSES[*]}")
RES=$(IFS=';' ; echo "${RES[*]}")

CLASSES_LENGTH=${#CLASSES[@]}

echo "[property]
gpu-id=0

onnx-file=$ONNX_FILENAME
model-engine-file=model.onnx_b1_gpu0_fp16.engine

gie-unique-id=4
net-scale-factor=0.00784313725490196
offsets=123.675;116.28;103.53

network-mode=2
batch-size=1

infer-dims=3;$RES
maintain-aspect-ratio=0
model-color-format=0 # 0: RGB # 1: GBR # 2: GRAY

network-type=2 # segmentation 
segmentation-output-order=0 # 0: NCHW # 1: NHWC
#segmentation-threshold=0.0
output-tensor-meta=1

num-detected-classes=$CLASSES_LENGTH
output-blob-names=output

[custom]
detected-classes=non-emtpy;$CLASSES
" > "$NVINFER_FILE"
