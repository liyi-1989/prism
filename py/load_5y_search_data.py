import pandas as pd
import datetime
import numpy as np
import os
import sys
sys.path.append('.')
from py.load_claim_data import load_claim_data

def load_5y_search_data(folder='0408'):
    base_dir = 'inst/extdata/search_data_5year'
    # weekly GT data is available to a max span of 5 years.
    searchTerms = os.listdir(os.path.join(base_dir,folder))

    for i in range(0, len(searchTerms)):
        search = pd.read_csv(os.path.join(base_dir,folder,searchTerms[i]))
        search['date'] = [datetime.datetime.strptime(t,'%Y-%m-%d').date() + datetime.timedelta(6) for t in search['date']]
        if i > 0:
            all_search = all_search.merge(search, on="date")
        else:
            all_search = search

    all_search = all_search.set_index('date')
    all_search.index.name = 'DATE'

    GT_startDate = np.min(search.date)
    GT_endDate = np.max(search.date)

    data_5y = load_claim_data(GT_startDate, GT_endDate)
    data_5y['allSearch'] = all_search

    return(data_5y)

if __name__ == "__main__":
    data_5y = load_5y_search_data()





