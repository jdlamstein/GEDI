#!/usr/bin/env bash


export GOOGLE_APPLICATION_CREDENTIALS="/Users/joshlamstein/Documents/GEDI/keys/GEDI-dcced49d5690.json"
echo $GOOGLE_APPLICATION_CREDENTIALS
# export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
# echo $PROJECT_ID
# export BUCKET_NAME="rebelbase"
# echo $BUCKET_NAME

export BUCKET_NAME="gs://rebelbase"
export REGION="us-central1"
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export JOB_NAME="gedi_${USER}_$(date +%Y%m%d_%H%M%S)" #--job-dir
# export JOB_NAME="flowers_joshualamstein_20181017_175120"
export GCS_PATH="${BUCKET_NAME}/${USER}/${JOB_NAME}"
export RUNTIME_VERSION=1.12
export MODEL_NAME=Gedi
export VERSION_NAME=v1
export PYTHON_VERSION=3.5
export X_TRAIN_PATH="${BUCKET_NAME}/GCS/train_Live.tfrecord"
export Y_TRAIN_PATH="${BUCKET_NAME}/GCS/train_Dead.tfrecord"
export X_VAL_PATH="${BUCKET_NAME}/GCS/val_Live.tfrecord"
export Y_VAL_PATH="${BUCKET_NAME}/GCS/val_Dead.tfrecord"
export X_TEST_PATH="${BUCKET_NAME}/GCS/test_Live.tfrecord"
export Y_TEST_PATH="${BUCKET_NAME}/GCS/test_Dead.tfrecord"

echo
echo "Using job id: " $JOB_NAME
#set -v -e

gcloud ml-engine jobs submit training $JOB_NAME --stream-logs --runtime-version $RUNTIME_VERSION --python-version 3.5 --job-dir $GCS_PATH --package-path trainer --module-name trainer.task --region $REGION --config config.yaml
