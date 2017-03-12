import subprocess

from setuptools import find_packages
from setuptools import setup
from setuptools.command.bdist_egg import bdist_egg as _bdist_egg


# Some custom command to run during setup. The command is not essential for this
# workflow. It is used here as an example. Each command will spawn a child
# process. Typically, these commands will include steps to install non-Python
# packages. For instance, to install a C++-based library libjpeg62 the following
# two commands will have to be added:
#
#     ['apt-get', 'update'],
#     ['apt-get', '--assume-yes', install', 'libjpeg62'],
#
# First, note that there is no need to use the sudo command because the setup
# script runs with appropriate access.
# Second, if apt-get tool is used then the first command needs to be 'apt-get
# update' so the tool refreshes itself and initializes links to download
# repositories.  Without this initial step the other apt-get install commands
# will fail with package not found errors. Note also --assume-yes option which
# shortcuts the interactive confirmation.
#
# The output of custom commands (including failures) will be logged in the
# worker-startup log.

CUSTOM_COMMANDS = [
    ['apt-get', 'update'],
    ['apt-get', '--assume-yes', 'install', 'libhdf5-dev'],
]


class CustomCommands(setuptools.Command):
    """A setuptools Command class able to run arbitrary commands."""
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def RunCustomCommand(self, command_list):
        print 'Running command: %s' % command_list
        p = subprocess.Popen(
            command_list,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # Can use communicate(input='y\n'.encode()) if the command run requires
        # some confirmation.
        stdout_data, _ = p.communicate()
        print 'Command output: %s' % stdout_data
        if p.returncode != 0:
            raise RuntimeError(
                'Command %s failed: exit code: %s' % (command_list, p.returncode))
        
    def run(self):
        for command in CUSTOM_COMMANDS:
            self.RunCustomCommand(command)
                
                
# Configure the required packages and scripts to install.
# Note that the Python Dataflow containers come with numpy already installed
# so this dependency will not trigger anything to be installed unless a version
# restriction is specified.


REQUIRED_PACKAGES = [
    "keras",
    "theano",
    "h5py",
]

setup(
    name='dataflow_preprocess',
    version='1.0',
    description='Image prepocessing code to be run on Google Dataflow',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    cmdclass={
        # Command class instantiated and run during easy_install scenarios.
        'bdist_egg': bdist_egg,
        'CustomCommands': CustomCommands,
    }
)
