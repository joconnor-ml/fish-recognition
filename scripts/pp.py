from dataflow_preprocess import dataflow_preprocess
import argparse

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

def default_args(argv):
  """Provides default values for Workflow flags."""
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
  parser.add_argument(
      '--project',
      type=str,
      help='The cloud project name to be used for running this pipeline')

  parser.add_argument(
      '--job_name',
      type=str,
      default='flowers-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
      help='A unique job identifier.')
  parser.add_argument(
      '--num_workers', default=20, type=int, help='The number of workers.')
  parser.add_argument('--cloud', default=False, action='store_true')
  parser.add_argument(
      '--runner',
      help='See Dataflow runners, may be blocking'
      ' or not, on cloud or not, etc.')

  parsed_args, _ = parser.parse_known_args(argv)

  if parsed_args.cloud:
    # Flags which need to be set for cloud runs.
    default_values = {
        'project':
            get_cloud_project(),
        'temp_location':
            os.path.join(os.path.dirname(parsed_args.output_path), 'temp'),
        'runner':
            'DataflowRunner',
        'extra_package':
            'gs://cloud-ml/sdk/cloudml-0.1.9-alpha.tar.gz',
        'save_main_session':
            True,
    }
  else:
    # Flags which need to be set for local runs.
    default_values = {
        'runner': 'DirectRunner',
    }

  for kk, vv in default_values.iteritems():
    if kk not in parsed_args or not vars(parsed_args)[kk]:
      vars(parsed_args)[kk] = vv

  return parsed_args


def main(argv):
  arg_dict = default_args(argv)
  run(arg_dict)


