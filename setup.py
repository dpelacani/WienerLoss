from setuptools import setup, find_packages


exclude = ['docs', 'tests', 'examples', 'performance']

setup(name='awloss',
      version="0.0.1",
      description="Pytorch implementation of the adaptive wiener loss",
      long_description="""
      Adaptive Wiener Loss provides a convolution-based method to compare data
      that can be used for several optimisation tasks or difference measures
      in general. It is based on the theory proposed in Adaptive Waveform
      Inversion (Warner and Guasch, 2014), and adapted to focus on deep
      learning tasks. It supports all operations from Pytorch and follows
      similar API """,
      project_urls={
          'Source Code': 'https://github.com/dekape/AWLoss',
          'Issue Tracker': 'https://github.com/dekape/AWLoss/issues',
      },
      url='https://github.com/dekape/AWLoss',
      platforms=["Linux", "Mac OS-X", "Unix"],
      test_suite='pytest',
      author="Deborah Pelacani Cruz",
      author_email='deborah.pelacani-cruz18@imperial.ac.uk',
      license='MIT',
      packages=find_packages(exclude=exclude))
