import pandas as pd
import numpy as np


# Generate a monte carlo sample from a master data set.  uses buckets for quasi-monte-carlo.  buckets=1 means
# traditional monte carlo.
def montecarlo_sample(data_frame: pd.DataFrame, buckets: int):
    samplesize = len(data_frame)
    bucketsize = max(1, int(samplesize / buckets))

    newdata = pd.DataFrame()
    x = 1
    while x < samplesize:
        for y in range(0, buckets):
            rangestart = y * bucketsize
            rangeend = min(samplesize, (y+1) * bucketsize)
            if rangeend <= rangestart:
                continue

            rndidx = np.random.randint(rangestart, rangeend)
            arow = data_frame.iloc[rndidx, ]
            newdata = newdata.append(arow, sort=False)
            x += 1
            if x >= samplesize:
                break
    return newdata
