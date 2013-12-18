from distutils.core import setup

setup(
    name='PyWRFPlus',
    version='0.1.0',
    author='John Lawson',
    author_email='john.rob.lawson@googlemail.com',
    packages=['PyWRFPlus'],
    scripts=['bin/casestudyexample.py'],
    url='http://pypi.python.org/pypi/PyWRFPlus/',
    license='LICENCE.txt',
    description='Pre- and post-processing of WRF data.',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.6.1",
        "meteogeneral >= 0.1.0"
    ],
)
