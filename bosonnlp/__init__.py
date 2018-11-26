# -*- coding: utf-8 -*-
"""
bosonnlp.py
===========

BosonNLP HTTP API 封装库（SDK）。

安装
----

:py:mod:`bosonnlp` 代码托管在 `GitHub`_，并且已经发布到 `PyPI`_，可以直接通过 `pip` 安装::

    $ pip install bosonnlp

:py:mod:`bosonnlp` 以 MIT 协议发布。

.. _GitHub: https://github.com/bosondata/bosonnlp.py
.. _PyPI: https://pypi.python.org/pypi/bosonnlp

使用教程
--------

    >>> from bosonnlp import BosonNLP
    >>> nlp = BosonNLP('YOUR_API_TOKEN')
    >>> nlp.sentiment('这家味道还不错')
    [[0.8758192096636473, 0.12418079033635264]]

可以在 `BosonNLP`_ 文档站点阅读详细的 BosonNLP HTTP API 文档。

.. _BosonNLP: http://docs.bosonnlp.com/
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import logging


__VERSION__ = '0.11.1'


from .client import BosonNLP, ClusterTask, CommentsTask
from .exceptions import HTTPError, TaskNotFoundError, TaskError, TimeoutError

# Set default logging handler to avoid "No handler found" warnings.
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logging.getLogger(__name__).addHandler(NullHandler())
