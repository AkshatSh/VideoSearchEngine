import numpy as np

def build_index(links):
    website_list = links.keys()
    return {website: index for index, website in enumerate(website_list)}
 
def build_transition_matrix(links, index):
    total_links = 0
    A = np.zeros((len(index), len(index)))
    for webpage in links:
        # dangling page
        if not links[webpage]:
            # Assign equal probabilities to transition to all the other pages
            A[index[webpage]] = np.ones(len(index)) / len(index)
        else:
            for dest_webpage in links[webpage]:
                total_links += 1
                A[index[webpage]][index[dest_webpage]] = 1.0 / len(links[webpage])
 
    return A

def pagerank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs((new_P - P).sum())
        if delta <= eps:
            return new_P
        P = new_P


links = {
    'vid1': set([vid2])
    'vid2': set([])
}
website_index = build_index(links)
A = build_transition_matrix(links, website_index)
results = pagerank(A)
