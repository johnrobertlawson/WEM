from distutils.core import setup

setup(
    name='postWRF',
    version='0.1.0',
    author='John Lawson',
    author_email='john.rob.lawson@googlemail.com',
    packages=['postWRF'],
    scripts=['bin/casestudyexample.py'],
    url='http://pypi.python.org/pypi/postWRF/',
    license='LICENCE.txt',
    description='Post-processing of WRF data.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.6.1",
    ],
)
