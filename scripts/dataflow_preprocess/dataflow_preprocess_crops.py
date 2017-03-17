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

from PIL import Image

from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
#from google.cloud.ml.io import SaveFeatures
import json

class ProcessImageBoxes(beam.DoFn):
  def process(self, element, size):
    from keras.preprocessing import image
    uri = element
    # TF will enable 'rb' in future versions, but until then, 'r' is
    # required.
    def _open_file_read_binary(uri):
      try:
        return file_io.FileIO(uri, mode='rb')
      except errors.InvalidArgumentError:
        return file_io.FileIO(uri, mode='r')

    try:
      with _open_file_read_binary(uri) as f:
        image_bytes = f.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        size_x, size_y = img.size
        img1 = img.crop((5, 5, size_x // 2 + 5, size_y // 2 + 5)).resize((size[1], size[0]))  # bottom left
        img2 = img.crop((size_x // 2 - 5, 5, size_x - 5, size_y // 2 + 5)).resize((size[1], size[0]))  # bottom right
        img3 = img.crop((5, size_y // 2 - 5, size_x // 2 + 5, size_y - 5)).resize((size[1], size[0]))  # top left
        img4 = img.crop((size_x // 2 - 5, size_y // 2 - 5, size_x - 5, size_y - 5)).resize((size[1], size[0]))  # top right

    # A variety of different calling libraries throw different exceptions here.
    # They all correspond to an unreadable file so we treat them equivalently.
    # pylint: disable broad-except
    except Exception as e:
      logging.exception('Error processing image %s: %s', uri, str(e))
      return

    fname = uri.split("/")[-1]
    yield fname, image.img_to_array(img1, "th"), image.img_to_array(img2, "th"), \
      image.img_to_array(img3, "th"), image.img_to_array(img4, "th"), size_x, size_y
    #x = self.image_data_generator.random_transform(x)
    #x = self.image_data_generator.standardize(x)


class ComputeFeatures(beam.DoFn):
  def __init__(self, size):
    self.size = size
    
  def start_bundle(self, context=None):
    os.environ["KERAS_BACKEND"] = "theano"
    import vgg16bn
    from keras import backend as K
    K.set_image_dim_ordering('th')
    logging.warn("Loading VGG")
    self.vgg = vgg16bn.Vgg16BN(include_top=False, size=self.size)
    logging.warn("Done loading VGG")

  def process(self, element):
    fname = element[0]
    logging.warn(element[1].shape)
    img1 = np.expand_dims(element[1], axis=0)
    img2 = np.expand_dims(element[2], axis=0)
    img3 = np.expand_dims(element[3], axis=0)
    img4 = np.expand_dims(element[4], axis=0)
    size_x = element[5]
    size_y = element[6]
    "Loads pre-built VGG model up to last convolutional layer"""
    try:
      emb1 = self.vgg.predict(img1)
      emb2 = self.vgg.predict(img2)
      emb3 = self.vgg.predict(img3)
      emb4 = self.vgg.predict(img4)
    except ValueError:
      emb1 = []
      emb2 = []
      emb3 = []
      emb4 = []
    yield json.dumps({
      "file": fname,
      "size_x": size_x,
      "size_y": size_y,
      "embedding1": emb1.tolist(),
      "embedding2": emb2.tolist(),
      "embedding3": emb3.tolist(),
      "embedding4": emb4.tolist(),
    })

  
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
      default=224,
      help='Target image X size in pixels')
  parser.add_argument(
      '--size_y',
      dest='size_y',
      default=448,
      help='Target image Y size in pixels')
  known_args, pipeline_args = parser.parse_known_args(argv)

    
  with beam.Pipeline(argv=pipeline_args) as p:
    read_input_source = beam.io.ReadFromText(
      known_args.input_path, strip_trailing_newlines=True, min_bundle_size=64)
    _ = (p
         | 'Read input' >> read_input_source
         | 'Process images' >> beam.ParDo(ProcessImageBoxes(), size=(known_args.size_y,
                                                                 known_args.size_x))
         | 'Compute features' >> beam.ParDo(ComputeFeatures(size=(known_args.size_y,
                                                                  known_args.size_x)))
         | 'save' >> beam.io.WriteToText(known_args.output_path)) 
         # | 'save' >> beam.io.avroio.WriteToAvro(known_args.output_path, schema)) 
         # | 'save' >> SaveFeatures(known_args.output_path)) 
