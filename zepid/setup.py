from setuptools import setup 

setup(name='zepid',version='0.1.0',description='Basic tool package for epidemiologic analyses',
      long_description='This package contains some essential epidemiological tools for epidemiology analyses in Python 3.x. Available'
      ' tools include summary measures (RR, OR, RD, NNT, IRD, IRR, PAF, ACR), these same measures for pandas '
      'dataframes, IC and ICR, effect measure plots, senstivity analysis tools, and inverse probability weights.',
      author='Paul Zivich',author_email='zivich.5@gmail.com',install_requires=['pandas',
      'numpy','statsmodels','matplotlib','scipy'])