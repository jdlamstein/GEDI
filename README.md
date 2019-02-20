GEDI machine learning for GCS (Google Cloud Storage).

# Setting up GCS

In order to run this on the cloud you will need:
1. Billing Account and Google Storage
2. Install the Cloud SDK
3. ML-Engine
4. Service Account Key

## Billing Account and Google Storage

The business billing account should be set up. If you're using a personal account, and this is your first time using GCS, you could get a free trial with $300 credit. 
  Go to https://cloud.google.com/storage/docs/quickstart-console
  For a personal account, click on Try Free in the top right corner. 
  
 Continue to follow the instructions in the link. 
 Create a project. 
 
 Enable Google Storage.
 Create a bucket. A bucket is where data is stored for machine learning. 
 
 ## Cloud SDK
 
In order to run scripts from the command line, you should install the Cloud SDK, https://cloud.google.com/sdk/docs/. 
The command gcloud is used to initiate machine learning. The command gsutil is used to copy files to the Google Storage Bucket. 

## ML-Engine

In addition storage, we need to enable the machine learning engine and connect it to billing. GCS has many services and you need to select which services will be enabled.
https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction

## Service Account Key

In order for GCS to recognize that your computer is connected to your billing account, you'll need to create a service account key, which is a json file. The link below shows how to do it. 
 https://cloud.google.com/iam/docs/creating-managing-service-account-keys
 
 # Running the GEDI script
 To train a model, the code is set up in a particular folder tree structure with a config.yaml file, and it's expected that the main function accept arguments from the command line. 
 
 The documentation is here: https://cloud.google.com/ml-engine/docs/tensorflow/how-tos
 
 ## Folder Structure
 The GCS portion of the git repo is under gcloud_training where you'll find a config.yaml file, setup.py, MANIFEST, and a folder marked 'trainer'.
  config.yaml - This file selects which gpus and cpus your model will run on. Descriptions of options in the yaml file may be found https://cloud.google.com/ml-engine/docs/tensorflow/machine-types
  setup.py - This file declares the package dependencies that GCS needs to install to run your script. 
  MANIFEST - Automatically generated, do not alter. 
  trainer - This folder contains the code. GCS expects the code to be in this folder. The main file that GCS will run is task.py, all others are dependencies. There must be an __init__.py file in order to run. 
  
 ## How to run
 Running bash script gedi_gcloud.sh by navigating to its directory in the command line and typing
 >> chmod u+x gedi_gcloud.sh
 >> . gedi_gcloud.sh
 
 The bash script includes the gcloud command to run the model at https://cloud.google.com/ml-engine/docs/tensorflow/training-jobs. 
 In the script, the command is:
 
 gcloud ml-engine jobs submit training $JOB_NAME --stream-logs --runtime-version $RUNTIME_VERSION --python-version 3.5 --job-dir $GCS_PATH --package-path trainer --module-name trainer.task --region $REGION --config config.yaml
 where 'gcloud ml-engine jobs submit training' announces we're launching a machine learning job. 
 The arguments are:
 
 $JOB_NAME - which is the name of this particular training. 
 --stream-logs - show logs from a running Cloud ML Engine job
 --runtime-version - Tensorflow version, currently uses 1.12.0
 --python-version - the python version
 --job-dir - Path to save model and logs. 
 --package-path - specifies the local path to the root directory of your application
 --module-name -  specifies the name of your application's main module
 --region - specifies region where GCS servers will run the model, should be region close to you
 
 # Description of Model
 
 This is a keras version of GEDI CNN that trains on top of Inception V3. So far, I have trained on ~20000 images using tfrecords, adding more images will require multiple tfrecords to size considerations and I'll need to adjust the data pipeline. I'm considering rewriting this in Tensorflow instead of keras. I use the tensorflow data api, so that data will be loaded in small portions with a data generator rather than loading all the data into the ml-engine at once. The computer doesn't have enough memory to load all the memory at once. Keras is built to run with Tensorflow and Theano and making Keras work with the TF Data API is relatively new because it's exclusive to tensorflow and not applicable to Theano. While keras works with TF data, tensorboard is throwing errors because keras writes the tensorboard file faster than GCS allows. So, I think tensorflow is better at the moment. 
  
  When you run this, you'll need to change filepaths, and connect the script to your own key. 

 
  
 
