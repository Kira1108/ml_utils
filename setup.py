from setuptools import setup
from setuptools import find_packages

setup(name='ml_utils',
      version='0.0.3',
      description='useful function for machine learning',
      author='The fastest man alive.',
      packages=find_packages(),
      install_requires=['numpy','tensorflow','matplotlib','click'],
      entry_points="""
        [console_scripts]
        mlxx=ml_utils.cli:cli
    """)