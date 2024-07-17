from setuptools import setup, find_packages


exclude = ['docs', 'tests', 'examples', 'performance']

setup(name='wiener-loss',
      version="0.0.2",
      description="Pytorch implementation of the Wienerloss",
      long_description="""
      Wiener Loss provides a convolution-based method to compare data
      that can be used for several optimisation tasks or difference measures
      in general as described in "Convolve and Conquer: Data Comparison
      with Wiener Filters" (Cruz et al, 2023). It supports all operations
      from Pytorch and follows similar API """,
      project_urls={
          'Source Code': 'https://github.com/dekape/WienerLoss',
          'Issue Tracker': 'https://github.com/dekape/WienerLoss/issues',
      },
      url='https://github.com/dekape/WienerLoss',
      platforms=["Linux", "Mac OS-X", "Unix"],
      test_suite='pytest',
      author="Deborah Pelacani Cruz",
      author_email='deborah.pelacani@gmail.com',
      license='MIT',
      packages=find_packages(exclude=exclude))
