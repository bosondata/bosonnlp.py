.. automodule:: bosonnlp

API
---

.. autoclass:: bosonnlp.BosonNLP
    :members:
    :member-order: bysource

.. autoclass:: bosonnlp.ClusterTask
   :members: push, analysis, status, wait_until_complete, result, clear

.. autoclass:: bosonnlp.CommentsTask
   :members: push, analysis, status, wait_until_complete, result, clear

Exceptions
----------

.. autoexception:: bosonnlp.HTTPError

.. autoexception:: bosonnlp.TaskNotFoundError

.. autoexception:: bosonnlp.TaskError

.. autoexception:: bosonnlp.TimeoutError
