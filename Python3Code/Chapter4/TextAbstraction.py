##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

from nltk import tokenize
from nltk import parse
from nltk.stem.snowball import SnowballStemmer
import unicodedata
import unidecode
from nltk.corpus import stopwords
import nltk
import math
import gensim
import gensim.models.ldamodel as lda

# This class includes a number of approaches that abstract text based data to structured features.
class TextAbstraction:

    col_name = 'words'
    bow = 'bow'

    # Tokenize the text: identify sentences and words within sentences. Returns a list of words.
    def tokenization(self, text):
        words = []
        sentences = tokenize.sent_tokenize(text)
        for sentence in sentences:
            words.extend(tokenize.word_tokenize(sentence))
        words = list(filter(lambda x: x != '.', words))
        return words

    # Create a clean set of words which are lower case and do not include any undesired characters.
    # Returns the cleaned set.
    def lower_case_and_filter_chars(self, words):
        new_words = []
        for word in words:
            # Take the lower case.
            word = word.lower()
            try:
                # Use the proper coding.
                word = word.decode('utf-8')
            except:
                word = word
                # something went wrong with the decoding, don't care for now.

            word = unidecode.unidecode(word)
            newText = ''

            # Select only the letters from the alphabet.
            for c in word:
                if ((c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c == ' '):
                    newText = newText + c
            if len(newText) > 0:
                new_words.append(newText)
        return new_words


    # Stem a list of words. Return the list of stemmed words.
    def stem(self, text):
        stemmer = SnowballStemmer("english")
        newText = []
        for w in text:
            newText.append(str(stemmer.stem(w)))
        return newText

    # Remove stopwords from a list of words. Returns the cleaned list.
    def remove_stop_words(self, text):
        stopwordList = stopwords.words('english')
        names = nltk.corpus.names.words()

        newText = []
        for w in text:
            if w.lower() not in stopwordList and w.lower() not in names:
                newText.append(w)
        return newText

    # Create combinations of <n> words for n-grams. Return a list of elements
    # that are the combination of <n> words that occur adjacent.
    def form_n_grams(self, words, n):
        n_grams = []
        for i in range(0, len(words)-n):
            n_grams.append('_'.join(words[i:i+n]))
        return n_grams

    # Identify words in the specified columns, create n-grams. Returns the same
    # data table with a column with text where each row is a list of cleaned n-grams
    # that occur in the cols of that row and the list of unique words is returned.
    def identify_words(self, data_table, cols, n):
        word_attributes = []

        # Create the cleaned text column.
        data_table[self.col_name] = 0
        data_table[self.col_name] = data_table[self.col_name].astype(object)

        # Pass all rows.
        for i in range(0, len(data_table.index)):
            words = []
            text = ''
            for col in cols:
                text = text + ' ' + data_table[col][i]

            # Perform the NLP pipeline.
            words = self.tokenization(text)
            lower_case_words = self.lower_case_and_filter_chars(words)
            stemmed_words = self.stem(lower_case_words)
            no_stopwords_words = self.remove_stop_words(stemmed_words)
            n_grams = self.form_n_grams(no_stopwords_words, n)
            current_set = set(word_attributes)
            new_set = set(n_grams)
            # Store the current set of n-grams found.
            word_attributes = list(current_set | new_set)
            # And add the found list of words to the table.
            data_table.set_value(i, self.col_name, n_grams)

        return data_table, word_attributes

    # Apply the bag of words approach upon the text that can be found in cols. It identifies
    # n-grams and creates columns for those n-grams. It computes the number of occurrences
    # of the n-grams per row and uses that as a value.
    def bag_of_words(self, data_table, cols, n):

        # Identify the words and clean the table.
        data_table, words = self.identify_words(data_table, cols, n)

        # Create columns for each word.
        for word in words:
            data_table[cols[0] + '_bow_' + word] = 0

            # And count the occurrences per row.
            for i in range(0, len(data_table.index)):
                data_table.iloc[i, data_table.columns.get_loc(f'{cols[0]}_bow_{word}')] = data_table[self.col_name][i].count(word)

        # Remove the temporary column we had created for the cleaned lists of words.
        del data_table[self.col_name]
        return data_table

    # Apply the bag of words approach upon the text that can be found in cols. It identifies
    # n-grams and creates columns for those n-grams. It computes the TF-IDF
    # of the n-grams per row and uses that as a value.
    def tf_idf(self, data_table, cols, n):

        # Identify the words and clean the table.
        data_table, words = self.identify_words(data_table, cols, n)

        # Create columns for each word.
        for word in words:
            data_table[cols[0] + '_tf_idf_' + word] = 0.0

            for i in range(0, len(data_table.index)):

                # And count the tf score.
                tf = data_table[self.col_name][i].count(word)
                data_table.iloc[i, data_table.columns.get_loc(f'{cols[0]}_tf_idf_{word}')] = tf

            # Compute the idf score over all rows.
            idf = math.log(float(len(data_table.index))/len(data_table.loc[data_table[cols[0] + '_tf_idf_' + word] > 0].index))
            # and multiply the rows with the idf.
            data_table[cols[0] + '_tf_idf_' + word] = data_table[cols[0] + '_tf_idf_' + word].mul(idf)
        # Remove the temporary column we had created for the cleaned lists of words.
        del data_table[self.col_name]
        return data_table

    # This function identifies n topics in the data using LDA and for each row computes a score for the
    # topic. It returns a dataset with columns added for the topics containing the scores per row.
    def topic_modeling(self, data_table, cols, n_topics):

        # Identify the words and clean the table.
        data_table, words = self.identify_words(data_table, cols)

        # Create a dictionary based on the words we have identified per row.
        dict_topics = gensim.corpora.Dictionary(data_table[self.col_name])
        # Create a corpus containing all words.
        corpus = [dict_topics.doc2bow([word]) for word in words]

        # Apply LDA.
        model = lda.LdaModel(corpus, id2word=dict_topics, num_topics=n_topics)

        # Get the topics we found.
        topics = model.show_topics(num_topics=n_topics, num_words=10, log=False, formatted=False)

        # Create columns for the topics.
        for topic in range(0, n_topics):
            data_table[f'{cols[0]}_topic_{topic}'] = 0.0

        # Score the topics per row and set the values accordingly.
        for i in range(0, len(data_table.index)):
            topic_scores = model[dict_topics.doc2bow(data_table[self.col_name][i])]
            for score in topic_scores:
                data_table.iloc[i, data_table.columns.get_loc(f'{cols[0]}_topic_{score[0]}')] = score[1]
        # Remove the temporary column we had created for the cleaned lists of words.
        del data_table[self.col_name]
        return data_table

