from setuptools import setup 

setup(name='zepid',version='0.1.0',description='Tool package for epidemiologic analyses',
      long_description='''This package contains some essential epidemiological tools for epidemiology analyses in Python 3.5+ Available 
      tools include basic association meaures, IC and ICR, effect measure plots, functional form assessments,
      senstivity analysis tools, and inverse probability weights.''',
      author='Paul Zivich',author_email='zepidpy@gmail.com',install_requires=['pandas',
      'numpy','statsmodels','matplotlib','scipy','networkx','lifelines','tabulate'])