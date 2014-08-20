from __future__ import absolute_import, division, print_function, unicode_literals

from setuptools import setup, find_packages


setup(
    name='bosonnlp',
    author='BosonData',
    author_email='support@bosondata.com.cn',
    description="BosonNLP.com API wrapper.",
    long_description=open('README.rst').read(),
    url='https://bosonnlp-py.readthedocs.org/',
    version='0.2.2',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'requests>=2.0.0',
    ],
    tests_require=[
        'pytest',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Testing',
    ]
)
