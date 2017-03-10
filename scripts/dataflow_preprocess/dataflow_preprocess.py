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

  
def run(argv=None):
  """Runs the pre-processing pipeline."""

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_path',
      required=True,
      help='Input specified as uri to CSV file. Each line of csv file '
      'contains colon-separated GCS uri to an image and labels.')
  parser.add_argument(
      '--output_path',
      required=True,
      help='Output directory to write results to.')
  parser.add_argument(
      '--size_x',
      dest='size_x',
      default=640,
      help='Target image X size in pixels')
  parser.add_argument(
      '--size_y',
      dest='size_y',
      default=360,
      help='Target image Y size in pixels')
  known_args, pipeline_args = parser.parse_known_args(argv)
  
  with beam.Pipeline(options=pipeline_args) as p:
    read_input_source = beam.io.ReadFromText(
      known_args.input_path, strip_trailing_newlines=True)
    _ = (p
         | 'Read input' >> read_input_source
         | 'Process images' >> beam.ParDo(ProcessImages(), size=(known_args.size_y,
                                                                 known_args.size_x))
         | 'Compute features' >> beam.ParDo(ComputeFeatures(size=(known_args.size_y,
                                                                  known_args.size_x)))
         #| 'save' >> beam.io.WriteToText('./test')) 
         | 'save' >> SaveFeatures(known_args.output_path)) 
