# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import gzip
import json
import uuid
import time
import datetime
from io import BytesIO
from functools import partial
import requests

from . import __VERSION__
from .exceptions import HTTPError, TaskNotFoundError, TaskError, TimeoutError


PY2 = sys.version_info[0] == 2
DEFAULT_BOSONNLP_URL = 'http://api.bosonnlp.com'
DEFAULT_TIMEOUT = 30 * 60


if PY2:
    text_type = unicode
    string_types = (str, unicode)
else:
    text_type = str
    string_types = (str,)


def _generate_id():
    return str(uuid.uuid4())


def _gzip_compress(buf):
    zbuf = BytesIO()
    with gzip.GzipFile(mode='wb', fileobj=zbuf, compresslevel=9) as zfile:
        zfile.write(buf)
    return zbuf.getvalue()


_json_dumps = partial(json.dumps, ensure_ascii=False, sort_keys=True)


class BosonNLP(object):
    """BosonNLP HTTP API 访问的封装类。

    :param string token: 用于 API 鉴权的 API Token。

    :param string bosonnlp_url: BosonNLP HTTP API 的 URL，默认为 `http://api.bosonnlp.com`。

    """

    def __init__(self, token, bosonnlp_url=DEFAULT_BOSONNLP_URL):
        self.token = token
        self.bosonnlp_url = bosonnlp_url

        # Enable keep-alive and connection-pooling.
        self.session = requests.session()
        self.session.headers['X-Token'] = token
        self.session.headers['Accept'] = 'application/json'
        self.session.headers['User-Agent'] = 'bosonnlp.py/{} {}'.format(
            __VERSION__, requests.utils.default_user_agent()
        )

    def _api_request(self, method, path, **kwargs):
        url = self.bosonnlp_url + path
        if method == 'POST':
            if 'data' in kwargs:
                headers = kwargs.get('headers', {})
                headers['Content-Type'] = 'application/json'
                data = _json_dumps(kwargs['data'])
                if isinstance(data, text_type):
                    data = data.encode('utf-8')
                if len(data) > 10 * 1024:  # 10K
                    headers['Content-Encoding'] = 'gzip'
                    data = _gzip_compress(data)
                kwargs['data'] = data
                kwargs['headers'] = headers

        r = self.session.request(method, url, **kwargs)

        http_error_msg = ''

        if 400 <= r.status_code < 600:
            reason = r.reason
            try:
                reason = r.json()['message']
            except:
                pass
            http_error_msg = 'HTTPError: %s %s' % (r.status_code, reason)

        if http_error_msg:
            raise HTTPError(http_error_msg, response=r)

        return r

    def sentiment(self, contents, news=False):
        """BosonNLP `情感分析接口 <http://docs.bosonnlp.com/sentiment.html>`_ 封装。

        :param contents: 需要做情感分析的文本或者文本序列。
        :type contents: string or sequence of string

        :param bool news: 默认为 :py:class:`False`，是否使用新闻语料训练的模型。

        :returns: 接口返回的结果列表。

        :raises: :py:exc:`~bosonnlp.HTTPError` 如果 API 请求发生错误。

        调用示例：

        >>> nlp = BosonNLP('YOUR_API_TOKEN')
        >>> nlp.sentiment('他是个傻逼')
        [[0.4464756252294154, 0.5535243747705846]]
        >>> nlp.sentiment(['他是个傻逼', '美好的世界'])
        [[0.4464756252294154, 0.5535243747705846], [0.6739600397344988, 0.3260399602655012]]
        """
        api_endpoint = '/sentiment/analysis'
        params = set()
        if news:
            params.add('news')
        if params:
            api_endpoint += ('?' + '&'.join(params))
        r = self._api_request('POST', api_endpoint, data=contents)
        return r.json()

    def convert_time(self, content, basetime=None):
        """BosonNLP `时间描述转换接口 <http://docs.bosonnlp.com/time.html>`_ 封装

        :param content: 中文时间描述字符串
        :type content: string

        :param basetime: 时间描述的基准时间，传入一个时间戳或datetime
        :type basetime: int or datetime.datetime

        :raises: :py:exc:`~bosonnlp.HTTPError` 如果 API 请求发生错误。

        :returns: 接口返回的结果
        """
        api_endpoint = '/time/analysis'
        params = {'pattern': content}
        if basetime:
            if isinstance(basetime, datetime.datetime):
                basetime = int(time.mktime(basetime.timetuple()))
            params['basetime'] = basetime
        r = self._api_request('POST', api_endpoint, params=params)
        return r.json()

    def classify(self, contents):
        """BosonNLP `新闻分类接口 <http://docs.bosonnlp.com/classify.html>`_ 封装。

        :param contents: 需要做分类的新闻文本或者文本序列。
        :type contents: string or sequence of string

        :returns: 接口返回的结果列表。

        :raises: :py:exc:`~bosonnlp.HTTPError` 如果 API 请求发生错误。

        调用示例：

        >>> nlp = BosonNLP('YOUR_API_TOKEN')
        >>> nlp.classify('俄否决安理会谴责叙军战机空袭阿勒颇平民')
        [5]
        >>> nlp.classify(['俄否决安理会谴责叙军战机空袭阿勒颇平民',
        ...               '邓紫棋谈男友林宥嘉：我觉得我比他唱得好',
        ...               'Facebook收购印度初创公司'])
        [5, 4, 8]
        """
        api_endpoint = '/classify/analysis'
        r = self._api_request('POST', api_endpoint, data=contents)
        return r.json()

    def suggest(self, word, top_k=None):
        """BosonNLP `语义联想接口 <http://docs.bosonnlp.com/suggest.html>`_ 封装。

        :param string word: 需要做语义联想的词。

        :param int top_k: 默认为 10，最大值可设定为 100。返回的结果条数。

        :returns: 接口返回的结果列表。

        :raises: :py:exc:`~bosonnlp.HTTPError` 如果 API 请求发生错误。

        调用示例：

        >>> nlp = BosonNLP('YOUR_API_TOKEN')
        >>> nlp.suggest('python', top_k=1)
        [[0.9999999999999992, 'python/x']]
        """
        api_endpoint = '/suggest/analysis'
        params = {}
        if top_k is not None:
            params['top_k'] = top_k
        r = self._api_request('POST', api_endpoint, params=params, data=word)
        return r.json()

    def extract_keywords(self, text, top_k=None, segmented=False):
        """BosonNLP `关键词提取接口 <http://docs.bosonnlp.com/keywords.html>`_ 封装。

        :param string text: 需要做关键词提取的文本。

        :param int top_k: 默认为 100，返回的结果条数。

        :param bool segmented: 默认为 :py:class:`False`，`text` 是否已进行了分词，如果为
            :py:class:`True`，则不会再对内容进行分词处理。

        :returns: 接口返回的结果列表。

        :raises: :py:exc:`~bosonnlp.HTTPError` 如果 API 请求发生错误。

        调用示例：

        >>> nlp = BosonNLP('YOUR_API_TOKEN')
        >>> nlp.extract_keywords('病毒式媒体网站：让新闻迅速蔓延', top_k=2)
        [[0.4580507649282757, '蔓延'], [0.44467176143180404, '病毒']]
        """
        api_endpoint = '/keywords/analysis'
        if segmented:
            api_endpoint += '?segmented'
        params = {}
        if top_k is not None:
            params['top_k'] = top_k
        r = self._api_request('POST', api_endpoint, params=params, data=text)
        return r.json()

    def depparser(self, contents):
        """BosonNLP `依存文法分析接口 <http://docs.bosonnlp.com/depparser.html>`_ 封装。

        :param contents: 需要做依存文法分析的文本或者文本序列。
        :type contents: string or sequence of string

        :returns: 接口返回的结果列表。

        :raises: :py:exc:`~bosonnlp.HTTPError` 如果 API 请求发生错误。

        调用示例：

        >>> nlp = BosonNLP('YOUR_API_TOKEN')
        >>> nlp.depparser('他是个傻逼')
        [{'role': ['SBJ', 'ROOT', 'NMOD', 'VMOD'],
          'head': [1, -1, 3, 1],
          'word': ['他', '是', '个', '傻逼'],
          'tag': ['PN', 'VC', 'M', 'NN']}]
        >>> nlp.depparser(['他是个傻逼', '美好的世界'])
        [{'role': ['SBJ', 'ROOT', 'NMOD', 'VMOD'],
          'head': [1, -1, 3, 1],
          'word': ['他', '是', '个', '傻逼'],
          'tag': ['PN', 'VC', 'M', 'NN']},
         {'role': ['DEC', 'NMOD', 'ROOT'],
          'head': [1, 2, -1],
          'word': ['美好', '的', '世界'],
          'tag': ['VA', 'DEC', 'NN']}]
        """
        api_endpoint = '/depparser/analysis'
        r = self._api_request('POST', api_endpoint, data=contents)
        return r.json()

    def ner(self, contents, sensitivity=None):
        """BosonNLP `命名实体识别接口 <http://docs.bosonnlp.com/ner.html>`_ 封装。

        :param contents: 需要做命名实体识别的文本或者文本序列。
        :type contents: string or sequence of string

        :param sensitivity: 准确率与召回率之间的平衡，
            设置成 1 能找到更多的实体，设置成 5 能以更高的精度寻找实体。
        :type sensitivity: int 默认为 3

        :returns: 接口返回的结果列表。

        :raises: :py:exc:`~bosonnlp.HTTPError` 如果 API 请求发生错误。

        调用示例：

        >>> nlp = BosonNLP('YOUR_API_TOKEN')
        >>> nlp.ner('成都商报记者 姚永忠')
        [{'entity': [[0, 2, 'product_name'], [3, 4, 'person_name']],
          'tag': ['ns', 'n', 'n', 'nr'],
          'word': ['成都', '商报', '记者', '姚永忠']}]
        >>> nlp.ner(['成都商报记者 姚永忠', '微软XP操作系统今日正式退休'])
        [{'entity': [[0, 2, 'product_name'], [3, 4, 'person_name']],
          'tag': ['ns', 'n', 'n', 'nr'],
          'word': ['成都', '商报', '记者', '姚永忠']},
         {'entity': [[0, 2, 'product_name'], [3, 4, 'time']],
          'tag': ['nt', 'x', 'nl', 't', 'ad', 'v'],
          'word': ['微软', 'XP', '操作系统', '今日', '正式', '退休']}]
        """
        api_endpoint = '/ner/analysis'
        params = {}
        if sensitivity is not None:
            params['sensitivity'] = sensitivity
        r = self._api_request('POST', api_endpoint, data=contents, params=params)
        return r.json()

    def tag(self, contents):
        """BosonNLP `分词与词性标注 <http://docs.bosonnlp.com/tag.html>`_ 封装。

        :param contents: 需要做分词与词性标注的文本或者文本序列。
        :type contents: string or sequence of string

        :returns: 接口返回的结果列表。

        :raises: :py:exc:`~bosonnlp.HTTPError` 如果 API 请求发生错误。

        调用示例：

        >>> nlp = BosonNLP('YOUR_API_TOKEN')
        >>> nlp.tag('成都商报记者 姚永忠')
        [{'tag': ['NR', 'NN', 'NN', 'NR'],
          'word': ['成都', '商报', '记者', '姚永忠']}]
        >>> nlp.tag(['成都商报记者 姚永忠', '微软XP操作系统今日正式退休'])
        [{'tag': ['NR', 'NN', 'NN', 'NR'],
          'word': ['成都', '商报', '记者', '姚永忠']},
         {'tag': ['NR', 'NN', 'NN', 'NN', 'NT', 'AD', 'VV'],
          'word': ['微软', 'XP', '操作', '系统', '今日', '正式', '退休']}]
        """
        api_endpoint = '/tag/analysis'
        r = self._api_request('POST', api_endpoint, data=contents)
        return r.json()

    def _cluster_push(self, task_id, contents):
        api_endpoint = '/cluster/push/' + task_id
        contents = ClusterTask._prepare_contents(contents)
        if not contents:
            return False
        for i in range(0, len(contents), 100):
            chunk = contents[i:i + 100]
            r = self._api_request('POST', api_endpoint, data=chunk)
        return r.ok

    def _cluster_analysis(self, task_id, alpha=None, beta=None):
        api_endpoint = '/cluster/analysis/' + task_id
        params = {}
        if alpha is not None:
            params['alpha'] = alpha
        if beta is not None:
            params['beta'] = beta
        r = self._api_request('GET', api_endpoint, params=params)
        return r.ok

    def _cluster_status(self, task_id):
        api_endpoint = '/cluster/status/' + task_id
        r = self._api_request('GET', api_endpoint)

        status = str(r.json()['status']).lower()

        if status == 'not found':
            raise TaskNotFoundError('cluster {} not found'.format(task_id), response=r)

        if status == 'error':
            raise TaskError('cluster {} error'.format(task_id), response=r)

        return status

    def _cluster_result(self, task_id):
        api_endpoint = '/cluster/result/' + task_id
        r = self._api_request('GET', api_endpoint)
        return r.json()

    def _cluster_clear(self, task_id):
        api_endpoint = '/cluster/clear/' + task_id
        r = self._api_request('GET', api_endpoint)
        return r.ok

    def cluster(self, contents, task_id=None, alpha=None, beta=None, timeout=DEFAULT_TIMEOUT):
        """BosonNLP `文本聚类接口 <http://docs.bosonnlp.com/cluster.html>`_ 封装。

        :param contents: 需要做文本聚类的文本序列或者 (_id, text) 序列或者
            {'_id': _id, 'text': text} 序列。如果没有指定 _id，则自动
            生成唯一的 _id。
        :type contents: sequence of string or sequence of (_id, text) or
            sequence of {'_id': _id, 'text': text}

        :param string task_id:
            唯一的 task_id，话题聚类任务的名字，可由字母和数字组成。

        :param float alpha: 默认为 0.8，聚类最大 cluster 大小

        :param float beta: 默认为 0.45，聚类平均 cluster 大小

        :param float timeout: 默认为 1800 秒（30 分钟），等待文本聚类任务完成的秒数。

        :returns: 接口返回的结果列表。

        :raises:

            :py:exc:`~bosonnlp.HTTPError` - 如果 API 请求发生错误

            :py:exc:`~bosonnlp.TaskError` - 任务出错。

            :py:exc:`~bosonnlp.TimeoutError` - 如果文本聚类任务未能在 `timeout` 时间内完成。

        调用示例：

        >>> nlp = BosonNLP('YOUR_API_TOKEN')
        >>> nlp.cluster(['今天天气好', '今天天气好', '今天天气不错', '点点楼头细雨',
        ...              '重重江外平湖', '当年戏马会东徐', '今日凄凉南浦'])
        [{'_id': 0, 'list': [0, 1], 'num': 2}]
        """
        if not contents:
            return []
        if isinstance(contents[0], string_types):
            contents = [{"_id": _id, "text": s} for _id, s in enumerate(contents)]
        cluster = None
        try:
            cluster = self.create_cluster_task(contents, task_id)
            cluster.analysis(alpha=alpha, beta=beta)
            cluster.wait_until_complete(timeout)
            result = cluster.result()
            return result
        finally:
            if cluster is not None:
                cluster.clear()

    def create_cluster_task(self, contents=None, task_id=None):
        """创建 :py:class:`~bosonnlp.ClusterTask` 对象。

        :param contents: 需要做典型意见的文本序列或者 (_id, text) 序列或者
            {'_id': _id, 'text': text} 序列。如果没有指定 _id，则自动
            生成唯一的 _id。默认为 :py:class:`None`。
        :type contents: :py:class:`None` or sequence of string or
            sequence of (_id, text) or sequence of {'_id': _id, 'text': text}

        :param string task_id: 默认为 :py:class:`None`，表示自动生成一个
            唯一的 task_id，典型意见任务的名字，可由字母和数字组成。

        :raises:

            :py:exc:`~bosonnlp.HTTPError` - 如果 API 请求发生错误
                如果 `contents` 不为 :py:class:`None`，则可能会调用
                :py:meth:`~bosonnlp.ClusterTask.push` 上传文本。

        :returns: :py:class:`~bosonnlp.ClusterTask` 实例。
        """
        return ClusterTask(self, contents, task_id)

    def _comments_push(self, task_id, contents):
        api_endpoint = '/comments/push/' + task_id
        contents = CommentsTask._prepare_contents(contents)
        if not contents:
            return False
        for i in range(0, len(contents), 100):
            chunk = contents[i:i + 100]
            r = self._api_request('POST', api_endpoint, data=chunk)
        return r.ok

    def _comments_analysis(self, task_id, alpha=None, beta=None):
        api_endpoint = '/comments/analysis/' + task_id
        params = {}
        if alpha is not None:
            params['alpha'] = alpha
        if beta is not None:
            params['beta'] = beta
        r = self._api_request('GET', api_endpoint, params=params)
        return r.ok

    def _comments_status(self, task_id):
        api_endpoint = '/comments/status/' + task_id
        r = self._api_request('GET', api_endpoint)

        status = str(r.json()['status']).lower()

        if status == 'not found':
            raise TaskNotFoundError('comments {} not found'.format(task_id), response=r)

        if status == 'error':
            raise TaskError('comments {} error'.format(task_id), response=r)

        return status

    def _comments_result(self, task_id):
        api_endpoint = '/comments/result/' + task_id
        r = self._api_request('GET', api_endpoint)
        return r.json()

    def _comments_clear(self, task_id):
        api_endpoint = '/comments/clear/' + task_id
        r = self._api_request('GET', api_endpoint)
        return r.ok

    def comments(self, contents, task_id=None, alpha=None, beta=None, timeout=DEFAULT_TIMEOUT):
        """BosonNLP `典型意见接口 <http://docs.bosonnlp.com/comments.html>`_ 封装。

        :param contents: 需要做典型意见的文本序列或者 (_id, text) 序列或者
            {'_id': _id, 'text': text} 序列。如果没有指定 _id，则自动
            生成唯一的 _id。
        :type contents: sequence of string or sequence of (_id, text) or
            sequence of {'_id': _id, 'text': text}

        :param string task_id: 默认为 :py:class:`None`，表示自动生成一个
            唯一的 task_id，典型意见任务的名字，可由字母和数字组成。

        :param float alpha: 默认为 0.8，聚类最大 cluster 大小

        :param float beta: 默认为 0.45，聚类平均 cluster 大小

        :param float timeout: 默认为 1800 秒（30 分钟），等待典型意见任务完成的秒数。

        :returns: 接口返回的结果列表。

        :raises:

            :py:exc:`~bosonnlp.HTTPError` - 如果 API 请求发生错误

            :py:exc:`~bosonnlp.TaskError` - 任务出错。

            :py:exc:`~bosonnlp.TimeoutError` - 如果典型意见任务未能在 `timeout` 时间内完成。

        调用示例：

        >>> nlp = BosonNLP('YOUR_API_TOKEN')
        >>> nlp.comments(['今天天气好', '今天天气好', '今天天气不错', '点点楼头细雨',
        ...               '重重江外平湖', '当年戏马会东徐', '今日凄凉南浦'] * 2)
        [{'_id': 0, 'list': [['点点楼头', 3], ['点点楼头', 10]],
          'num': 2, 'opinion': '点点楼头'},
         {'_id': 1, 'list': [['重重江外', 4], ['重重江外', 11]],
          'num': 2, 'opinion': '重重江外'},
         {'_id': 2, 'list': [['当年戏马', 5], ['当年戏马', 12]],
          'num': 2, 'opinion': '当年戏马'},
         {'_id': 3, 'list': [['今日凄凉', 6], ['今日凄凉', 13]],
          'num': 2, 'opinion': '今日凄凉'}]
        """
        if not contents:
            return []
        if isinstance(contents[0], string_types):
            contents = [{"_id": _id, "text": s} for _id, s in enumerate(contents)]
        comments = None
        try:
            comments = self.create_comments_task(contents, task_id)
            comments.analysis(alpha=alpha, beta=beta)
            comments.wait_until_complete(timeout)
            result = comments.result()
            return result
        finally:
            if comments is not None:
                comments.clear()

    def create_comments_task(self, contents=None, task_id=None):
        """创建 :py:class:`~bosonnlp.CommentsTask` 对象。

        :param contents: 需要做典型意见的文本序列或者 (_id, text) 序列或者
            {'_id': _id, 'text': text} 序列。如果没有指定 _id，则自动
            生成唯一的 _id。默认为 :py:class:`None`。
        :type contents: :py:class:`None` or sequence of string or
            sequence of (_id, text) or sequence of {'_id': _id, 'text': text}

        :param string task_id: 默认为 :py:class:`None`，表示自动生成一个
            唯一的 task_id，典型意见任务的名字，可由字母和数字组成。

        :raises:

            :py:exc:`~bosonnlp.HTTPError` - 如果 API 请求发生错误
                如果 `contents` 不为 :py:class:`None`，则可能会调用
                :py:meth:`~bosonnlp.CommentsTask.push` 上传文本。

        :returns: :py:class:`~bosonnlp.CommentsTask` 实例。
        """
        return CommentsTask(self, contents, task_id)


class _ClusterTask(object):

    def __init__(self, nlp, contents=None, task_id=None):
        if task_id is None:
            task_id = _generate_id()

        self.task_id = task_id
        self._contents = []

    @staticmethod
    def _prepare_contents(contents):
        if not contents:
            return []
        if isinstance(contents[0], string_types):
            contents = [{"_id": _generate_id(), "text": s} for s in contents]
        elif isinstance(contents[0], tuple):
            contents = [{"_id": _id, "text": s} for _id, s in contents]
        return contents

    def push(self, contents):
        """批量上传需要进行处理的文本。

        :param contents: 需要处理的的文本序列或者 (_id, text) 序列或者
            {'_id': _id, 'text': text} 序列。如果没有指定 _id，则自动
            生成唯一的 _id。
        :type contents: sequence of string or sequence of (_id, text) or
            sequence of {'_id': _id, 'text': text}

        :raises: :py:exc:`~bosonnlp.HTTPError` - 如果 API 请求发生错误
        """
        contents = self._prepare_contents(contents)
        if self._push(contents):
            self._contents.extend(contents)

    def analysis(self, alpha=None, beta=None):
        """启动分析任务

        :param float alpha: 默认为 0.8，最大 cluster 大小

        :param float beta: 默认为 0.45，平均 cluster 大小

        :raises: :py:exc:`~bosonnlp.HTTPError` - 如果 API 请求发生错误
        """
        return self._analysis(alpha=alpha, beta=beta)

    def wait_until_complete(self, timeout=None):
        """等待任务完成。

        :param float timeout: 等待任务完成的秒数，默认为 :py:class:`None`，
            表示不会超时。

        :raises:

            :py:exc:`~bosonnlp.HTTPError` - 如果 API 请求发生错误

            :py:exc:`~bosonnlp.TaskNotFoundError` - 任务不存在。

            :py:exc:`~bosonnlp.TaskError` - 任务出错。

            :py:exc:`~bosonnlp.TimeoutError` - 如果任务未能在 `timeout` 时间内完成。
        """
        elapsed = 0.0
        seconds_to_sleep = 1.0
        i = 0
        while True:
            time.sleep(seconds_to_sleep)

            status = self.status()
            if status == "done":
                return

            elapsed += seconds_to_sleep
            if timeout and elapsed >= timeout:
                raise TimeoutError('{0!r} timed out'.format(self))

            i = i + 1
            if i % 3 == 0 and seconds_to_sleep < 64:
                seconds_to_sleep = seconds_to_sleep + seconds_to_sleep

    def status(self):
        """获取任务状态。

        :returns: 任务状态。

            +---------------+------------------------------+
            |状态           |说明                          |
            +---------------+------------------------------+
            | ``received``  |成功接收到分析请求            |
            +---------------+------------------------------+
            | ``running``   |数据分析正在进行中            |
            +---------------+------------------------------+
            | ``done``      |分析已完成                    |
            +---------------+------------------------------+

        :raises:

            :py:exc:`~bosonnlp.HTTPError` - 如果 API 请求发生错误

            :py:exc:`~bosonnlp.TaskNotFoundError` - 任务不存在。

            :py:exc:`~bosonnlp.TaskError` - 任务出错。
        """
        return self._status()

    def result(self):
        """返回任务的结果。

        :returns: 接口返回的结果。

        :raises: :py:exc:`~bosonnlp.HTTPError` - 如果 API 请求发生错误
        """
        return self._result()

    def clear(self):
        """清空服务器端缓存的文本和结果。

        :returns: 是否成功。

        :raises: :py:exc:`~bosonnlp.HTTPError` - 如果 API 请求发生错误
        """
        return self._clear()

    def __repr__(self):
        return "<{0.__class__.__name__} {0.task_id}>".format(self)


class ClusterTask(_ClusterTask):
    """文本聚类任务封装类。

    一般通过 :py:meth:`~bosonnlp.BosonNLP.create_cluster_task` 创建实例。

    >>> nlp = BosonNLP('YOUR_API_TOKEN')
    >>> cluster = nlp.create_cluster_task()

    之后可以多次调用 :py:meth:`~bosonnlp.ClusterTask.push` 上传文本。

    >>> cluster.push(['今天天气好', '今天天气好', '今天天气不错', '点点楼头细雨'])
    >>> cluster.push(['重重江外平湖'])
    >>> cluster.push(['当年戏马会东徐', '今日凄凉南浦'])

    然后调用 :py:meth:`~bosonnlp.ClusterTask.analysis` 启动文本聚类任务。

    >>> cluster.analysis()

    一般文本聚类任务都用于处理大量的文本，需要比较长的处理时间。可以调用
    :py:meth:`~bosonnlp.ClusterTask.wait_until_complete` 等待任务完成。

    >>> cluster.wait_until_complete(2 * 60)  # 最多等待 2 分钟

    也可以调用 :py:meth:`~bosonnlp.ClusterTask.status` 查看任务状态。

    >>> cluster.status()
    'done'

    如果任务成功完成，可以通过 :py:meth:`~bosonnlp.ClusterTask.result` 获取结果。

    >>> cluster.result()
    [{'_id': '9e90c56e-f1bb-4605-b995-304af733207a',
      'list': ['9e90c56e-f1bb-4605-b995-304af733207a',
               'a3feff6b-6d1b-4f46-a2d8-0eea25c7f17f'],
      'num': 2}]

    获取结果后，可以通过 :py:meth:`~bosonnlp.ClusterTask.clear` 清空服务器端缓存的文本和结果。

    >>> cluster.clear()
    True

    :param nlp: :py:class:`~bosonnlp.BosonNLP` 类实例。
        其他参数和 :py:meth:`~bosonnlp.BosonNLP.cluster` 一致。
    """
    def __init__(self, nlp, contents=None, task_id=None):
        super(ClusterTask, self).__init__(nlp, contents, task_id)

        self._push = partial(nlp._cluster_push, self.task_id)
        self._analysis = partial(nlp._cluster_analysis, self.task_id)
        self._status = partial(nlp._cluster_status, self.task_id)
        self._result = partial(nlp._cluster_result, self.task_id)
        self._clear = partial(nlp._cluster_clear, self.task_id)

        self.push(contents)


class CommentsTask(_ClusterTask):
    """典型意见任务封装类。

    一般通过 :py:meth:`~bosonnlp.BosonNLP.create_comments_task` 创建实例。

    >>> nlp = BosonNLP('YOUR_API_TOKEN')
    >>> comments = nlp.create_comments_task()

    之后可以多次调用 :py:meth:`~bosonnlp.CommentsTask.push` 上传文本。

    >>> comments.push(['今天天气好', '今天天气好', '今天天气不错', '点点楼头细雨'] * 2)
    >>> comments.push(['重重江外平湖'] * 2)
    >>> comments.push(['当年戏马会东徐', '今日凄凉南浦'] * 2)

    然后调用 :py:meth:`~bosonnlp.CommentsTask.analysis` 启动典型意见任务。

    >>> comments.analysis()

    一般典型意见任务都用于处理大量的文本，需要比较长的处理时间。可以调用
    :py:meth:`~bosonnlp.CommentsTask.wait_until_complete` 等待任务完成。

    >>> comments.wait_until_complete(2 * 60)  # 最多等待 2 分钟

    也可以调用 :py:meth:`~bosonnlp.CommentsTask.status` 查看任务状态。

    >>> comments.status()
    'done'

    如果任务成功完成，可以通过 :py:meth:`~bosonnlp.CommentsTask.result` 获取结果。

    >>> comments.result()
    [{'_id': 0,
      'list': [['点点楼头', '19c248e3-605b-4785-8785-ccd2d1b034cc'],
               ['点点楼头', '576d1d08-ff02-4bc5-9edf-fbc7a6915e44']],
      'num': 2,
      'opinion': '点点楼头'},
     {'_id': 2,
      'list': [['重重江外', 'a75e24bd-8597-4865-8254-2e9cab229770'],
               ['重重江外', '47db4d92-6328-45cd-98f8-099691e82c07']],
      'num': 2,
      'opinion': '重重江外'},
     {'_id': 4,
      'list': [['当年戏马', 'da0cd13f-4f13-4476-a541-6214df3b4dd9'],
               ['当年戏马', '89aecf45-4b78-4522-9ed2-ea76ed552f24']],
      'num': 2,
      'opinion': '当年戏马'},
     {'_id': 5,
      'list': [['今日凄凉', 'a5c1f6a9-b6a6-4877-b073-0b59bc67fa48'],
               ['今日凄凉', '9d38ecab-5860-44e9-9e3c-9ad4d4ed3547']],
      'num': 2,
      'opinion': '今日凄凉'}]

    获取结果后，可以通过 :py:meth:`~bosonnlp.CommentsTask.clear` 清空服务器端缓存的文本和结果。

    >>> comments.clear()
    True

    :param nlp: :py:class:`~bosonnlp.BosonNLP` 类实例。
        其他参数和 :py:meth:`~bosonnlp.BosonNLP.comments` 一致。
    """
    def __init__(self, nlp, contents=None, task_id=None):
        super(CommentsTask, self).__init__(nlp, contents, task_id)

        self._push = partial(nlp._comments_push, self.task_id)
        self._analysis = partial(nlp._comments_analysis, self.task_id)
        self._status = partial(nlp._comments_status, self.task_id)
        self._result = partial(nlp._comments_result, self.task_id)
        self._clear = partial(nlp._comments_clear, self.task_id)

        self.push(contents)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
