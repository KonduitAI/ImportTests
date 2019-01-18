from setuptools import setup
from setuptools import find_packages

setup(name='tfoptests',
      description='Generate, persist and load tensorflow graphs',
      url='https://github.com/deeplearning4j/TFOpTests',
      install_requires=['numpy', 'tensorflow', 'Pillow'],
      packages=find_packages(),
      zip_safe=False)
