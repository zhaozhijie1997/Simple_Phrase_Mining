import os
import re
import json
import unicodedata
from functools import wraps
import html
import string
from urllib.parse import unquote
import pkg_resources

import plane

PUNCTUATION_WHITELIST = ['-', '.', '+', '&']
LANGUAGES = ['TW', 'EN', 'VN', 'NUM', 'TH', 'PT','ES']
TAGS = ['html', 'url', 'email', 'space']

POLISH = plane.build_new_regex('POLISH', 
    r'[a-zA-Z\u0104\u0105\u0106\u0107\u0118\u0119\u0141\u0142\u0143\u0144\xd3\xf3\u015a\u015b\u0179\u017a\u017b\u017c]+')

def verbose(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self = func(self, *args, **kwargs)
        if self.verbose:
            print('[DEBUG] After {:20} => "{}"'.format(
                func.__name__, self._text))
        return self
    return wrapper

def replace_func(_text):
    _text =  _text.group(1) + " " + _text.group(3)
    return _text

class Polish:
    """text preprocessing.

    :param bool verbose: activate DEBUG model
    :param dict norm_map: use customized punctuation normalization mappings
    :param str segmenter: use custumized segmenter
    """
    def __init__(self, verbose=False, norm_map=None, segmenter=None):
        self._text = ''
        self.verbose = verbose
        self.punc = plane.Punctuation(norm_map) if norm_map else plane.punc
        self.plane = plane.Plane()
        self.segmenter = segmenter
        self.british2american = None
        self.tag_map = {
            'html': plane.HTML,
            'email': plane.EMAIL,
            'space': plane.SPACE,
            'url': plane.URL,
        }
        self.lang = {
            'PL': POLISH,
            'TW': plane.CHINESE,
            'VN': plane.VIETNAMESE,
            'TH': plane.THAI,
            'EN': plane.ENGLISH,
            'PT': plane.BraSCII,
            'ES': plane.BraSCII,
            'NUM': plane.NUMBER,
        }
        self.pattern1 = re.compile(r"""([a-z]+)(-)([a-z])""", re.VERBOSE)
        self.pattern2 = re.compile(r"""([a-z]+)(\.)([a-z])""", re.VERBOSE)

    def update(self, text):
        self._text = text
        return self

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text

    def init_english_map(self, file=None):
        if not file:
            file = '/ldap_home/chiman.wong/git/kevin/british_american.json'

        with open(file) as f:
            self.british2american = json.load(f)

    @verbose
    def american(self):
        """Transfer British English to American English
        """
        if not self.british2american:
            self.init_english_map()
        self._text = ' '.join([self.british2american.get(word.lower(), word)
                               for word in self._text.split(' ')])
        return self

    @verbose
    def normalize_unicode(self, form='NFC'):
        """Unicode Normalization.

        :param str form: Unicode format, 'NFC', 'NFKC', 'NFD', 'NFKD'

        For more information:

        - http://unicode.org/reports/tr15/
        - https://docs.python.org/3.7/library/unicodedata.html
        - https://unicode.org/charts/normalization/
        """
        self._text = unicodedata.normalize(form, self._text)
        return self


    @verbose
    def normalize_punctuation(self):
        """Transfer punctuations from other languages to English punctuations.
        """
        self._text = self.punc.normalize(self._text)
        return self
    
    @verbose
    def remove_special_punctuation(self):
        self._text = self.pattern1.sub(replace_func, self._text)
        self._text = self.pattern2.sub(replace_func, self._text)
        return self

    @verbose
    def remove_punctuation(self, whitelist=None):
        """Remove all the punctuations belongs to Unicode::Category::[P]

        - https://www.compart.com/en/unicode/category/Po
        """
        if whitelist is None:
            whitelist = PUNCTUATION_WHITELIST

        remove_punc = [item for item in string.punctuation if item not in whitelist]
        self._text = "".join([item if item not in remove_punc else " " for item in self._text])
        self._text = " ".join(self._text.split())
        return self

    @verbose
    def unescape_html(self):
        """HTML Escape

        - https://dev.w3.org/html5/html-author/charref
        - https://www.freeformatter.com/html-entities.html
        """
        self._text = html.unescape(self._text)
        return self

    @verbose
    def unquote_url(self):
        """URL unquote
        """
        self._text = unquote(self._text)
        return self

    @verbose
    def filter_lang(self, langs=None):
        """Extract specific language characters from text.

        :param list[str] lang: ['TW', 'EN', 'VN', 'TH', 'NUM']
        """
        if langs is None:
            langs = LANGUAGES

        for lang in langs:
            if lang not in self.lang:
                raise NameError('Unknown lang: {}. Only support {}.'.format(
                    lang, self.lang.keys()))

        regex = sum([self.lang.get(lang) for lang in langs] + [plane.SPACE])
        self._text = ''.join([t.value for t in
                              plane.extract(self._text, regex)])
        return self

    @verbose
    def remove_tag(self, tags=None):
        """The order of tags matters.
        :param list[str] tag: ['html', 'url', 'email', 'space']
        """
        if tags is None:
            tags = TAGS

        for t in tags:
            if t not in self.tag_map:
                raise NameError('Unknown tag: {}. Only support {}.'.format(
                    t, self.tag_map.keys()))

            self._text = plane.replace(self._text, self.tag_map[t])
        return self

    @verbose
    def segment(self, region='PL'):
        self._text = self._text.split()
        return self

    def lower(self, region='PL'):
        self._text = self._text.lower()
        return self

    def preprocess(self, text, lang='PL'):
        """All-in-one method. It's suitable for the common circumstance

        :param str text: text
        :param str country: ['TW', 'TH', 'SG', 'MY', 'ID', 'VN', 'PH']
        """
        if lang in ['PL', 'TW', 'VN', 'TH', 'PT','ES']:
            langs = [lang, 'EN', 'NUM']
        else:
            langs = ['EN', 'NUM']

        return (self.update(text)
                .normalize_unicode()
                .unescape_html()
                .american()
                .filter_lang(langs)
                .remove_tag()
                .remove_special_punctuation()
                .remove_punctuation()
                .lower()
                .text
                )






