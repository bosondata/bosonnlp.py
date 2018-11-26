# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import gzip
import json
import logging
import uuid
import time
import datetime
from io import BytesIO
from functools import partial
import requests

from . import __VERSION__
from .exceptions import HTTPError, TaskNotFoundError, TaskError, TimeoutError


PY2 = sys.version_info[0] == 2
DEFAULT_BOSONNLP_URL = 'https://api.bosonnlp.com'
DEFAULT_TIMEOUT = 30 * 60


if PY2:
    text_type = unicode
    string_types = (str, unicode)
else:
    text_type = str
    string_types = (str,)


logger = logging.getLogger(__name__)


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

    :param string bosonnlp_url: BosonNLP HTTP API 的 URL，默认为 `https://api.bosonnlp.com`。

    :param bool compress: 是否压缩大于 10K 的请求体，默认为 True。

    :param int timeout: HTTP 请求超时时间，默认为 60 秒。

    """

    def __init__(self, token, bosonnlp_url=DEFAULT_BOSONNLP_URL, compress=True, session=None, timeout=60):
        self.token = token
        self.bosonnlp_url = bosonnlp_url.rstrip('/')
        self.compress = compress
        self.timeout = timeout

        # Enable keep-alive and connection-pooling.
        self.session = session or requests.session()
        self.session.headers['X-Token'] = token
        self.session.headers['Accept'] = 'application/json'
        self.session.headers['User-Agent'] = 'bosonnlp.py/{} {}'.format(
            __VERSION__, requests.utils.default_user_agent()
        )

    def _api_request(self, method, path, **kwargs):
        kwargs.setdefault('timeout', self.timeout)
        url = self.bosonnlp_url + path
        if method == 'POST':
            if 'data' in kwargs:
                headers = kwargs.get('headers', {})
                headers['Content-Type'] = 'application/json'
                data = _json_dumps(kwargs['data'])
                if isinstance(data, text_type):
                    data = data.encode('utf-8')
                if len(data) > 10 * 1024 and self.compress:  # 10K
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

    def sentiment(self, contents, model='general'):
        """BosonNLP `情感分析接口 <http://docs.bosonnlp.com/sentiment.html>`_ 封装。

        :param contents: 需要做情感分析的文本或者文本序列。
        :type contents: string or sequence of string

        :param model: 使用不同语料训练的模型，默认使用通用模型。
        :type model: string

        :returns: 接口返回的结果列表。

        :raises: :py:exc:`~bosonnlp.HTTPError` 如果 API 请求发生错误。

        调用示例：

        >>> import os
        >>> nlp = BosonNLP(os.environ['BOSON_API_TOKEN'])
        >>> nlp.sentiment('这家味道还不错', model='food')
        [[0.9991737012037423, 0.0008262987962577828]]
        >>> nlp.sentiment(['这家味道还不错', '菜品太少了而且还不新鲜'], model='food')
        [[0.9991737012037423, 0.0008262987962577828],
         [9.940036427291687e-08, 0.9999999005996357]]
        """
        api_endpoint = '/sentiment/analysis?' + model
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

        调用示例：

        >>> import os
        >>> nlp = BosonNLP(os.environ['BOSON_API_TOKEN'])
        >>> _json_dumps(nlp.convert_time("2013年二月二十八日下午四点三十分二十九秒"))
        '{"timestamp": "2013-02-28 16:30:29", "type": "timestamp"}'
        >>> import datetime
        >>> _json_dumps(nlp.convert_time("今天晚上8点到明天下午3点", datetime.datetime(2015, 9, 1)))
        '{"timespan": ["2015-09-02 20:00:00", "2015-09-03 15:00:00"], "type": "timespan_0"}'

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

        >>> import os
        >>> nlp = BosonNLP(os.environ['BOSON_API_TOKEN'])
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

        >>> import os
        >>> nlp = BosonNLP(os.environ['BOSON_API_TOKEN'])
        >>> nlp.suggest('北京', top_k=2)
        [[1.0, '北京/ns'], [0.7493540460397998, '上海/ns']]
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

        >>> import os
        >>> nlp = BosonNLP(os.environ['BOSON_API_TOKEN'])
        >>> nlp.extract_keywords('病毒式媒体网站：让新闻迅速蔓延', top_k=2)
        [[0.8391345017584958, '病毒式'], [0.3802418301341705, '蔓延']]
        """
        api_endpoint = '/keywords/analysis'
        params = {}
        if segmented:
            params['segmented'] = 1
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

        >>> import os
        >>> nlp = BosonNLP(os.environ['BOSON_API_TOKEN'])
        >>> nlp.depparser('今天天气好')
        [{'head': [2, 2, -1],
          'role': ['TMP', 'SBJ', 'ROOT'],
          'tag': ['NT', 'NN', 'VA'],
          'word': ['今天', '天气', '好']}]
        >>> nlp.depparser(['今天天气好', '美好的世界'])
        [{'head': [2, 2, -1],
          'role': ['TMP', 'SBJ', 'ROOT'],
          'tag': ['NT', 'NN', 'VA'],
          'word': ['今天', '天气', '好']},
         {'head': [1, 2, -1],
          'role': ['DEC', 'NMOD', 'ROOT'],
          'tag': ['VA', 'DEC', 'NN'],
          'word': ['美好', '的', '世界']}]
        """
        api_endpoint = '/depparser/analysis'
        r = self._api_request('POST', api_endpoint, data=contents)
        return r.json()

    def ner(self, contents, sensitivity=None, segmented=False, space_mode='3'):
        """BosonNLP `命名实体识别接口 <http://docs.bosonnlp.com/ner.html>`_ 封装。

        :param contents: 需要做命名实体识别的文本或者文本序列。
        :type contents: string or sequence of string

        :param sensitivity: 准确率与召回率之间的平衡，
            设置成 1 能找到更多的实体，设置成 5 能以更高的精度寻找实体。
        :type sensitivity: int 默认为 3

        :param segmented: 输入是否为分词结果
        :type segmented: boolean 默认为 False

        :param space_mode: 分词空格保留选项
        :type space_mode: int（整型）, 0-3有效，默认为 3

        :returns: 接口返回的结果列表。

        :raises: :py:exc:`~bosonnlp.HTTPError` 如果 API 请求发生错误。

        调用示例：

        >>> import os
        >>> nlp = BosonNLP(os.environ['BOSON_API_TOKEN'])
        >>> nlp.ner('成都商报记者 姚永忠', sensitivity=2)
        [{'entity': [[0, 2, 'product_name'],
                     [2, 3, 'job_title'],
                     [3, 4, 'person_name']],
          'tag': ['ns', 'n', 'n', 'nr'],
          'word': ['成都', '商报', '记者', '姚永忠']}]

        >>> nlp.ner(['成都商报记者 姚永忠', '微软XP操作系统今日正式退休'])
        [{'entity': [[0, 2, 'product_name'],
                     [2, 3, 'job_title'],
                     [3, 4, 'person_name']],
          'tag': ['ns', 'n', 'n', 'nr'],
          'word': ['成都', '商报', '记者', '姚永忠']},
         {'entity': [[0, 2, 'product_name'],
                     [3, 4, 'time']],
          'tag': ['nz', 'nx', 'nl', 't', 'ad', 'v'],
          'word': ['微软', 'XP', '操作系统', '今日', '正式', '退休']}]
        """
        api_endpoint = '/ner/analysis'
        params = {'space_mode': space_mode}
        if sensitivity is not None:
            params['sensitivity'] = sensitivity
        if segmented:
            params['segmented'] = True

        r = self._api_request('POST', api_endpoint, data=contents, params=params)
        return r.json()

    def tag(self, contents, space_mode=0, oov_level=3, t2s=0, special_char_conv=0):
        """BosonNLP `分词与词性标注 <http://docs.bosonnlp.com/tag.html>`_ 封装。

        :param contents: 需要做分词与词性标注的文本或者文本序列。
        :type contents: string or sequence of string

        :param space_mode: 空格保留选项
        :type space_mode: int（整型）, 0-3有效

        :param oov_level: 枚举强度选项
        :type oov_level:  int（整型）, 0-4有效

        :param t2s: 繁简转换选项，繁转简或不转换
        :type t2s:  int（整型）, 0-1有效

        :param special_char_conv: 特殊字符转化选项，针对回车、Tab等特殊字符转化或者不转化
        :type special_char_conv:  int（整型）, 0-1有效

        :returns: 接口返回的结果列表。

        :raises: :py:exc:`~bosonnlp.HTTPError` 如果 API 请求发生错误。

        调用参数及返回值详细说明见：http://docs.bosonnlp.com/tag.html

        调用示例：

        >>> import os
        >>> nlp = BosonNLP(os.environ['BOSON_API_TOKEN'])

        >>> result = nlp.tag('成都商报记者 姚永忠')
        >>> _json_dumps(result)
        '[{"tag": ["ns", "n", "n", "nr"], "word": ["成都", "商报", "记者", "姚永忠"]}]'

        >>> format_tag_result = lambda tagged: ' '.join('%s/%s' % x for x in zip(tagged['word'], tagged['tag']))
        >>> result = nlp.tag("成都商报记者 姚永忠")
        >>> format_tag_result(result[0])
        '成都/ns 商报/n 记者/n 姚永忠/nr'

        >>> result = nlp.tag("成都商报记者 姚永忠", space_mode=2)
        >>> format_tag_result(result[0])
        '成都/ns 商报/n 记者/n  /w 姚永忠/nr'

        >>> result = nlp.tag(['亚投行意向创始成员国确定为57个', '“流量贵”频被吐槽'], oov_level=0)
        >>> format_tag_result(result[0])
        '亚/ns 投/v 行/n 意向/n 创始/vi 成员国/n 确定/v 为/v 57/m 个/q'

        >>> format_tag_result(result[1])
        '“/wyz 流量/n 贵/a ”/wyy 频/d 被/pbei 吐槽/v'
        """
        api_endpoint = '/tag/analysis'
        params = {
            'space_mode': space_mode,
            'oov_level': oov_level,
            't2s': t2s,
            'special_char_conv': special_char_conv,
        }
        r = self._api_request('POST', api_endpoint, params=params, data=contents)
        return r.json()

    def summary(self, title, content, word_limit=0.3, not_exceed=False):
        """BosonNLP `新闻摘要 <http://docs.bosonnlp.com/summary.html>`_ 封装。

        :param title: 需要做摘要的新闻标题。如果没有标题，请传空字符串。
        :type title: unicode

        :param content: 需要做摘要的新闻正文。
        :type content: unicode

        :param word_limit: 摘要字数限制。
            当为 float 时，表示字数为原本的百分比，0.0-1.0有效；
            当为 int 时，表示摘要字数。

            .. note::

               传 1 默认为百分比。

        :type word_limit: float or int

        :param not_exceed: 是否严格限制字数。
        :type not_exceed: bool，默认为 False

        :returns: 摘要。

        :raises: :py:exc:`~bosonnlp.HTTPError` 当API请求发生错误。

        调用参数及返回值详细说明见： http://docs.bosonnlp.com/summary.html

        调用示例：

        >>> import os
        >>> nlp = BosonNLP(os.environ['BOSON_API_TOKEN'])

        >>> content = (
                '腾讯科技讯（刘亚澜）10月22日消息，前优酷土豆技术副总裁'
                '黄冬已于日前正式加盟芒果TV，出任CTO一职。'
                '资料显示，黄冬历任土豆网技术副总裁、优酷土豆集团产品'
                '技术副总裁等职务，曾主持设计、运营过优酷土豆多个'
                '大型高容量产品和系统。'
                '此番加入芒果TV或与芒果TV计划自主研发智能硬件OS有关。')
        >>> title = '前优酷土豆技术副总裁黄冬加盟芒果TV任CTO'

        >>> nlp.summary(title, content, 0.1)
        腾讯科技讯（刘亚澜）10月22日消息，前优酷土豆技术副总裁黄冬已于日前正式加盟芒果TV，出任CTO一职。
        """
        api_endpoint = '/summary/analysis'

        not_exceed = int(not_exceed)

        data = {
            'not_exceed': not_exceed,
            'percentage': word_limit,
            'title': title,
            'content': content
        }

        r = self._api_request('POST', api_endpoint, data=data)
        return r.json()

    def _cluster_push(self, task_id, contents):
        api_endpoint = '/cluster/push/' + task_id
        contents = ClusterTask._prepare_contents(contents)
        if not contents:
            return False
        for i in range(0, len(contents), 100):
            chunk = contents[i:i + 100]
            r = self._api_request('POST', api_endpoint, data=chunk)
            logger.info('Pushed %d of %d documents for clustering.' % (i+len(chunk), len(contents)))
        return r.ok

    def _cluster_analysis(self, task_id, alpha=None, beta=None):
        api_endpoint = '/cluster/analysis/' + task_id
        params = {}
        if alpha is not None:
            params['alpha'] = alpha
        if beta is not None:
            params['beta'] = beta
        r = self._api_request('GET', api_endpoint, params=params)
        logger.info('Cluster analysis started.')
        return r.ok

    def _cluster_status(self, task_id):
        api_endpoint = '/cluster/status/' + task_id
        r = self._api_request('GET', api_endpoint)

        status = str(r.json()['status']).lower()

        if status == 'not found':
            raise TaskNotFoundError('cluster {} not found'.format(task_id), response=r)

        if status == 'error':
            raise TaskError('cluster {} error'.format(task_id), response=r)

        logger.info('Status: %s.' % status)
        return status

    def _cluster_result(self, task_id):
        api_endpoint = '/cluster/result/' + task_id
        v = self._api_request('GET', api_endpoint).json()

        logger.info('%d comments fetched.' % len(v))
        return v

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

        >>> import os
        >>> nlp = BosonNLP(os.environ['BOSON_API_TOKEN'])
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
            logger.info('Pushed %d of %d documents for comment clustering.' % (i+len(chunk), len(contents)))
        return r.ok

    def _comments_analysis(self, task_id, alpha=None, beta=None):
        api_endpoint = '/comments/analysis/' + task_id
        params = {}
        if alpha is not None:
            params['alpha'] = alpha
        if beta is not None:
            params['beta'] = beta
        r = self._api_request('GET', api_endpoint, params=params)
        logger.info('Comments analysis started.')
        return r.ok

    def _comments_status(self, task_id):
        api_endpoint = '/comments/status/' + task_id
        r = self._api_request('GET', api_endpoint)

        status = str(r.json()['status']).lower()

        if status == 'not found':
            raise TaskNotFoundError('comments {} not found'.format(task_id), response=r)

        if status == 'error':
            raise TaskError('comments {} error'.format(task_id), response=r)

        logger.info('Status: %s.' % status)
        return status

    def _comments_result(self, task_id):
        api_endpoint = '/comments/result/' + task_id
        v = self._api_request('GET', api_endpoint).json()

        logger.info('%d comments fetched.' % len(v))
        return v

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

        >>> import os
        >>> nlp = BosonNLP(os.environ['BOSON_API_TOKEN'])
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
        if timeout is not None:
            seconds_to_sleep = min(seconds_to_sleep, timeout)
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

    >>> import os
    >>> nlp = BosonNLP(os.environ['BOSON_API_TOKEN'])
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

    >>> import os
    >>> nlp = BosonNLP(os.environ['BOSON_API_TOKEN'])
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
