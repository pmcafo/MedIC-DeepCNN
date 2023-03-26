from setuptools import setup
from setuptools import find_packages

setup(
   name='ensmic',
   version='1.0',
   description='An analysis on Ensemble Learning optimized Neural Network Classification',
   url='https://github.com/muellerdo/ensmic',
   author='Dominik Müller',
   author_email='dominik.mueller@informatik.uni-augsburg.de',
   license='GPLv3',
   long_description="An analysis on Ensemble Learning optimized Neural Network Classification",
   long_description_content_type="text/markdown",
   packages=find_packages(),
   install_requires=['tensorflow>=2.4.0',
                     'aucmedi>=0.3.0',
                     'pandas>=1.1.4',
                     '