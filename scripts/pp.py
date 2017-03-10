from dataflow_preprocess import dataflow_preprocess
import sys


def main(argv):
  dataflow_preprocess.run()


if __name__ == '__main__':
  main(sys.argv[1:])
