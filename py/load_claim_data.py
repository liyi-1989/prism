import pandas as pd
import datetime
import numpy as np
import os

def load_claim_data(GT_startDate = datetime.date(2004,1,3),
                    GT_endDate = datetime.date(2016,12,31),
                    data_path = "inst/extdata/ICNSA.csv"
                    ):
    icnsa = pd.read_csv(data_path)
    icnsa['DATE'] = [datetime.datetime.strptime(t,'%Y-%m-%d').date() for t in icnsa['DATE']]
    claim_all = icnsa.set_index('DATE')
    claim_all.columns = ['icnsa']

    # early data prior to GT data is available
    startDate = datetime.date(1980,1,5)
    startIdx = np.where(claim_all.index == startDate)[0][0]
    endIdx = np.where(claim_all.index == GT_startDate)[0][0]
    claim_earlyData=claim_all[startIdx:endIdx]

    # claim data when GT data is available
    startIdx = np.where(claim_all.index == GT_startDate)[0][0]
    endIdx = np.where(claim_all.index == GT_endDate)[0][0] + 1
    claim_data=claim_all[startIdx:endIdx]

    data_all = {}
    data_all['claim_earlyData'] = claim_earlyData
    data_all['claim_all'] = claim_all
    data_all['claim_data'] = claim_data

    return(data_all)


if __name__ == "__main__":
    df = load_claim_data()



