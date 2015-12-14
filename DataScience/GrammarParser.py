''' https://gist.github.com/alexbowe/879414#file-nltk-intro-py-L34'''
from nltk.corpus import stopwords
import nltk

stopwords = stopwords.words('english')
lemmatizer = nltk.WordNetLemmatizer()
stemmer_alt = nltk.stem.porter.PorterStemmer()

# Used when tokenizing words
sentence_re = r'''(?x)      # set flag to allow verbose regexps
      ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
    | \w+(-\w+)*            # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
    | \.\.\.                # ellipsis
    | [][.,;"'?():-_`]      # these are separate tokens
'''
# This grammar is from: S. N. Kim, T. Baldwin, and M.-Y. Kan.
# Evaluating n-gram based evaluation metrics for automatic keyphrase extraction.
# Technical report, University of Melbourne, Melbourne 2010.

grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
"""


class GrammarParser(object):
    """Fancier preprocessing of corpus using a grammar."""

    def leaves(self, tree):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in tree.subtrees(filter=lambda t:t.label() == 'NP'):
            yield subtree.leaves()

    def normalise(self, word):
        """Normalises words to lowercase and stems and lemmatizes it."""
        word = word.lower()
        word = stemmer_alt.stem_word(word)
        word = lemmatizer.lemmatize(word)

        return word

    def acceptable_word(self, word):
        """Checks conditions for acceptable word: length, stopword."""
        accepted = bool(2 <= len(word) <= 20
            and word.lower() not in stopwords)

        return accepted

    def get_terms(self, tree):
        """Filters the main tree and it's subtrees for 'leaves', normalizes the
        words in the leaves and returns a generator."""

        for leaf in self.leaves(tree):
            term = [ self.normalise(w) for w,_ in leaf if self.acceptable_word(w) ]

            yield term

    def get_words(self, terms):
        """Loops over the terms and returns a single string of the words."""

        out = []
        for term in terms:
            for word in term:
                out.append(word)

        return " ".join(out)

    def main(self, text):
        """Breaks a single string into a tree using the grammar and returns
        the specified words as a string."""

        if text is None:
            return None

        chunker = nltk.RegexpParser(grammar)

        toks = nltk.regexp_tokenize(text, sentence_re)
        postoks = nltk.tag.pos_tag(toks)

        #print postoks
        tree = chunker.parse(postoks)

        terms = self.get_terms(tree)

        words = self.get_words(terms)

        return words
