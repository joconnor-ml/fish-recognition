# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Example dataflow pipeline for preparing image training data.
"""
import argparse
import csv
import datetime
import io
import logging
import os
import subprocess
import sys
import numpy as np

sys.setrecursionlimit(10000)

import apache_beam as beam
from apache_beam.metrics import Metrics
try:
  from apache_beam.utils.pipeline_options import PipelineOptions
except ImportError:
  from apache_beam.utils.options import PipelineOptions

from google.cloud.ml.io import SaveFeatures

def configure_pipeline(p, opt):
  """Specify PCollection and transformations in pipeline."""
  read_input_source = beam.io.ReadFromText(
      opt.input_path, strip_trailing_newlines=True)
  _ = (p
       | 'Read input' >> read_input_source
       | 'Process images' >> beam.ParDo(ProcessImages(), size=(opt.size_y, opt.size_x))
       | 'Compute features' >> beam.ParDo(ComputeFeatures(size=(opt.size_y, opt.size_x)))
       #| 'save' >> beam.io.WriteToText('./test')) 
       | 'save' >> SaveFeatures(opt.output_path)) 

       
class ProcessImages(beam.DoFn):
  def process(self, element, size):
    from keras.preprocessing import image
    x = image.img_to_array(image.load_img(element, target_size=size))
    x = self.image_data_generator.random_transform(x)
    x = self.image_data_generator.standardize(x)

    
class ComputeFeatures(beam.DoFn):
  def process(self, element, vgg):
    import vgg16bn
    "Loads pre-built VGG model up to last convolutional layer"""
    vgg = vgg16bn.Vgg16BN(include_top=False, size=size)
    yield vgg.predict(np.expand_dims(element, axis=0))

  
#def save_features(data):
#  import dl_utils
#  features = data
#  dl_utils.save_array("cnn_features.dat", features)


def run(in_args=None):
  """Runs the pre-processing pipeline."""
  pipeline_options = PipelineOptions.from_dictionary(vars(in_args))
  with beam.Pipeline(options=pipeline_options) as p:
    configure_pipeline(p, in_args)

  
def get_cloud_project():
  cmd = ['gcloud', 'config', 'list', 'project', '--format=value(core.project)']
  with open(os.devnull, 'w') as dev_null:
    return subprocess.check_output(cmd, stderr=dev_null).strip()


if __name__ == '__main__':
  main(sys.argv[1:])
