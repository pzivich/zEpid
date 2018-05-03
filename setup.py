from setuptools import setup 


setup(name='zepid',
    version='0.1.0',
    description='Tool package for epidemiologic analyses',
    author='Paul Zivich',
    author_email='zepidpy@gmail.com',
    url = 'https://github.com/pzivich/zepid',
    classifiers = ['Programming Language :: Python :: 3.5']
    install_requires=['pandas'>=0.18,'numpy','statsmodels','matplotlib','scipy','networkx','lifelines','tabulate'],
    keywords=['epidemiology'])
