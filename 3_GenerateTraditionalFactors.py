from pandas_datareader import data as pdr
import datetime
import yfinance as yf
import pickle
import pandas as pd
import numpy as np

with open('Alpha_all.pickle', 'rb') as f:
    Alpha_all = pickle.load(f)

Return_all = Alpha_all[list(Alpha_all.keys())[6]]

def extraction(stocks, start, end):
    def data(ticker):
        return(pdr.get_data_yahoo(ticker,
                                  start = start,
                                  end = end)
              )
    FAANG_stock = map(data, stocks)
    return(pd.concat(FAANG_stock,
                     keys = stocks,
                     names = ["Company", "Date"]
                    )
          )

stocks = ["^GSPC", "^DJI", "^IXIC", # equity
          "GC=F", "CL=F", # commodity
          "^TNX"] # bond

FAANG = extraction(stocks,
                   datetime.datetime(2015, 4, 21),
                   datetime.datetime(2022, 6, 7)
                  )

Daily_Closing_Prices = FAANG[["Adj Close"]].reset_index().pivot("Date",
                                                                "Company",
                                                                "Adj Close")

TraditionMacroFactorsIndex = pd.DataFrame(index = pd.to_datetime(Return_all.index))

TraditionMacroFactors = TraditionMacroFactorsIndex.merge(Daily_Closing_Prices, left_index = True, right_index=True, how = "left")

TraditionMacroFactors.fillna(method = "ffill", inplace = True)

df2 = pd.DataFrame(Daily_Closing_Prices.iloc[0,:]).T
TraditionMacroFactors = pd.concat([df2, TraditionMacroFactors])

TraditionMacroFactors = np.log(TraditionMacroFactors/TraditionMacroFactors.shift(1)).dropna()

TraditionMacroFactors = TraditionMacroFactors.merge(TraditionMacroFactorsIndex, left_index=True, right_index=True,how = "outer")
TraditionMacroFactors.fillna(0, inplace = True)

TraditionMacroFactors.to_csv("TraditionMacroFactors.csv", index = True)