from setuptools import setup
setup(
   name='SapiatFinalProject',
   version='0.1.0',
   author='SCU Team ',
   author_email='aac@scu.edu',
   packages=['dataanalysis','optimalportfolio','dataprep'],
   url='http://pypi.python.org/pypi/PackageName/',
   license='LICENSE.txt',
   description='SAPiat Regime Change Project',
   long_description=open('README.txt').read(),
   install_requires=[
                    "pandas",
                     "numpy",
                     "time",
                     "rpy2 = 2.9.8",
                     "pathlib"
   ],
)