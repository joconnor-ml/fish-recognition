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
    uri, height, width, x, y = element.split(",")
    try:
      height = int(height)
      width = int(width)
      x = int(x)
      y = int(y)
    except ValueError:
      return
    if height == 0 and width == 0:  # no fish
      return
    # TF will enable 'rb' in future versions, but until then, 'r' is
    # required.
    def _open_file_read_binary(uri):
      try:
        return file_io.FileIO(uri, mode='rb')
      except errors.InvalidArgumentError:
        return file_io.FileIO(uri, mode='r')

    def range_overlap(a_min, a_max, b_min, b_max):
      '''Neither range is completely greater than the other
      '''
      return (a_min <= b_max) and (b_min <= a_max)
    def overlap(x1, width1, y1, height1, x2, width2, y2, height2):
      '''Overlapping rectangles overlap both horizontally & vertically
      '''
      return range_overlap(x1, x1+width1, x2, x2+width2) and range_overlap(y1, y1+height1, y2, y2+height2)

    try:
      with _open_file_read_binary(uri) as f:
        image_bytes = f.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        size_x, size_y = img.size
        for i in range(12):
          x2 = np.random.uniform(0,0.8) * size_x
          y2 = np.random.uniform(0,0.8) * size_y
          width2 = np.random.uniform(0.05,0.4) * size_x
          height2 = np.random.uniform(0.05,0.4) * size_y
          overlapping = overlap(x, width, y, height, x2, width2, y2, height2)
          contained = x2 + width2 < size_x and y2 + height2 < size_y
          if contained and not overlapping:
            break
          if i > 10: return
          
        img = img.crop((x2, y2, x2 + width2, y2 + height2))
        img = img.resize((size[1], size[0]))
                

    # A variety of different calling libraries throw different exceptions here.
    # They all correspond to an unreadable file so we treat them equivalently.
    # pylint: disable broad-except
    except Exception as e:
      logging.exception('Error processing image %s: %s', uri, str(e))
      return

    fname = uri.split("/")[-1]
    logging.warn("{} {} {} {}".format(fname, image.img_to_array(img, "th")[0], size_x, size_y))
    yield fname, image.img_to_array(img, "th"), size_x, size_y
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
    img = np.expand_dims(element[1], axis=0)
    size_x = element[2]
    size_y = element[3]
    "Loads pre-built VGG model up to last convolutional layer"""
    try:
      emb = self.vgg.predict(img)
    except ValueError:
      emb = []
    yield json.dumps({
      "file": fname,
      "size_x": size_x,
      "size_y": size_y,
      "embedding": emb.tolist()
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
      default=299,
      help='Target image X size in pixels')
  parser.add_argument(
      '--size_y',
      dest='size_y',
      default=299,
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
