from database_utils import get_all_data
import re
from collections import OrderedDict 
# Main file for ranking the summaries

# Main API is here, more files may be used for the implementation

# returns dictionary of form, sorted in descending order by count
# {
#   count: [entry, entry,...]
# }


def rank_pages(query):
    '''
    Rank the summaries based on the search query, and return a list sorted by rank
    '''
    # get summaries somehow

    terms = query.split()
    data = get_all_data()

    # For testing
    # data = [{
    #     "name" : " test 1",
    #     "summary": "hello dog",
    #     "url": ""}, 
    #     {
    #     "name" : " test 4",
    #     "summary": "dog dog",
    #     "url": ""}, 
    #     {
    #     "name" : " test 2",
    #     "summary": "hell",
    #      "url": ""
    #     }, {
    #     "name" : " test 3",
    #     "summary": "hello hello dog",
    #     "url": ""
    #     }]
    
    result = {}


    for entry in data:
        summary = entry['summary']
      
        count = 0
        for term in terms:
            count += len(re.findall(term, summary))
        if count > 0:
            if count in result:
                result[count].append(entry)
            else: 
                result[count] = [entry]

    return OrderedDict(sorted(result.items(), reverse=True))
