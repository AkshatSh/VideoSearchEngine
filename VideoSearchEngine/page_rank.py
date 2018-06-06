from sklearn.feature_extraction.text import TfidfVectorizer
from database_utils import get_all_data
from collections import OrderedDict 
import operator 

def rank_pages(summaries, query):
    vect = TfidfVectorizer()
    result = {}

    for video in summaries:
        tfidf = vect.fit_transform([video['summary'], query])
        score = (tfidf * tfidf.T).A[1][0] 

        if(score > 0.2):
            result[video['name']] = score

    return OrderedDict(sorted(result.items(), key=operator.itemgetter(1), reverse=True))
