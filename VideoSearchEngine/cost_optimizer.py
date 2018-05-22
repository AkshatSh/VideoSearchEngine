import numpy
import math

'''
Let n be number of workers
Let x be number of frames
Let c be the cost of network_io for frames f
let s be the cost of the summary for frames f

If we were to distribute our system, the total cost would be:

c * n + (x/n) * s

We want to minimize this, so lets take the partial derivative
with respect to n

c + xs (-1/(n^2)) 

Find critical points by setting to 0

0 = c + xs (-1/(n^2)) 

-c =  xs (-1/(n^2)) 

-c/xs = (-1/(n^2)) 

c/xs = 1 / n^2

n^2 = xs / c

n = sqrt(xs / c)

So if we use the number of workers to be sqrt(xs / c), we
will have the optimal number of workers

'''



def estimate_best_num_workers(frames):
    x = float(frames)
    s = float(estimate_cost_of_summary(frames))
    c = float(estimate_cost_of_distribution(frames))
    return math.sqrt( ( x * s) / c)

def estimate_cost_of_summary(frames):
    pass

def estimate_cost_of_distribution(frames):
    pass

def cost_of_join(workers, frames_per_worker):
    pass