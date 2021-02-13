# The following code strips accents and separate words from punctuation
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.special import softmax
from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=FutureWarning)
import unicodedata, nltk


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


import re


def preprocess_sentence(w):
    w = unicode_to_ascii(w.strip().lower())

    # creating a space between a word and the punctuation following it eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping
    # -punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^\w?.!,¿]+", " ", w)
    w = re.sub(r"([?.!,¿])", "", w)
    w = w.strip()

    return w


class StopWordTransformer(BaseEstimator, TransformerMixin):
    """ An example transformer that returns the element-wise square root..

    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    input_shape : tuple
        The shape the data passed to :meth:`fit`
    """

    def __init__(self, singletons=True, threshold=None, exclusion_ratio=0.02):

        self.freq_dict = {}
        self.stopwords = []
        self.vocab = list()
        self.idf_dict = {}
        self.singletons = singletons
        self.threshold = threshold
        self.ratio = exclusion_ratio
        self.tf_matrix = np.array([[]])

    def calculate_idf(self, count):
        return np.log2(self.X_.size / count)

    def find_in_document(self, term):
        count = 0
        for document in self.X_:
            if re.search(re.escape(term), document):
                count += 1
        return term, self.calculate_idf(count)

    def count_term_in_doc(self):

        p = Pool(5)
        self.idf_dict = dict(p.map(self.find_in_document, self.vocab))
        return self.idf_dict

    def get_idf_dict(self):
        return self.idf_dict

    def get_tf_dict(self):
        return self.freq_dict

    def add_high_tf(self):

        counter = CountVectorizer()

        # Calculating the tf array
        self.tf_matrix = counter.fit_transform(self.X_).toarray()

        # Adding high frequency terms to stopwords
        ordered_idx = np.argsort(self.tf_matrix.sum(axis=0))[:-(self.threshold + 1):-1]

        self.stopwords.extend(counter.get_feature_names()[ordered_idx].tolist())

    def calculate_threshold(self, ratio=0.01):

        threshold = int(len(' '.join(self.X_).split()) * ratio)
        if threshold is 0:
            threshold = 1
        return threshold

    def add_low_idf(self):

        tfidf = TfidfVectorizer()

        tfidf_matrix = tfidf.fit_transform(self.tf_matrix).toarray()
        idf_array = tfidf_matrix.sum(axis=0)/self.tf_matrix.sum(axis=0)
        ordered_idx = np.argsort(idf_array)[:-(self.threshold + 1):-1]

        self.stopwords.extend(tfidf.get_feature_names()[ordered_idx].tolist())

    def add_singleton(self):
        self.stopwords.extend([word for word, freq in
                               Counter(' '.join(self.X_).split()).items() if freq == 1])

    def get_stopwords(self):
        return self.stopwords

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = np.array(X)
        self.input_shape_ = X.shape
        self.y_ = y

        self.vocab = set(' '.join(X).split())

        self.X_ = X

        if not self.threshold:
            self.threshold = self.calculate_threshold()

        self.add_high_tf()
        if self.singletons:
            self.add_singleton()
        self.add_low_idf()
        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The array containing the element-wise square roots of the values
            in `X`
        """
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])

        # Input validation
        X = np.array(X, dtype=str).reshape(-1, 1).copy()
        X = check_array(X)

        remove_stopwords = lambda doc: ' '.join([word for word in doc[0].split() if word not in set(self.stopwords)])
        X = np.array(list(map(remove_stopwords, X)))

        return X
