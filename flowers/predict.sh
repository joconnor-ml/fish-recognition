#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This sample assumes you're already setup for using CloudML.  If this is your
# first time with the service, start here:
# https://cloud.google.com/ml/docs/how-tos/getting-set-up

# Now that we are set up, we can start processing some flowers images.
declare -r PROJECT=fishing-160312
declare -r JOB_ID="fish_train_${USER}_$(date +%Y%m%d_%H%M%S)"
declare -r BUCKET="gs://fish_bucket"
declare -r GCS_PATH="${BUCKET}/${USER}/fish_phuhem_20170302_222624"
declare -r DICT_FILE=gs://fish_bucket/dict.txt

echo
echo "Using job id: " $JOB_ID
set -v -e

# Training on CloudML is quick after preprocessing.  If you ran the above
# commands asynchronously, make sure they have completed before calling this one.
gcloud beta ml jobs submit training "$JOB_ID" \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  -- \
  --output_path "${GCS_PATH}/predictions" \
  --eval_data_paths "${GCS_PATH}/preproc/test*" \
  --label_count=8 \
  --write_predictions
