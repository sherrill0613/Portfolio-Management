import pandas as pd
import json
import requests
import ssl
from datetime import datetime, timezone, timedelta
import time
import sqlite3

url = "https://api.exchange.coinbase.com/products"

headers = {"Accept": "application/json"}

response = requests.get(url, headers=headers)

message = response.json()
names = []

for data in message:
    contract_name = data["id"]
    names.append(contract_name)

# remove some abnormal value
remove = ['SUSHI-USD', 'INDEX-USDT', 'INDEX-USD', "WAMPL-USD", "WAMPL-USDT"]
for stk in remove:
    names.pop(names.index(stk))

# change the ssl state so we do not see an error for https:// content
ssl._create_default_https_context = ssl._create_unverified_context


def getUnixTimestamp(dateString, timespec='milliseconds', pattern=None):
    """
        This function takes a specific date string and tries to turn a
        date pattern into a unix timestamp (epoch time). It will return a unix timestamp.
    """
    if pattern is None:
        pattern = "%Y-%m-%dT%H:%M:%S"
    try:
        unix = int(time.mktime(time.strptime(dateString, pattern)))
    except ValueError as ve:
        iso_str = dateString.astimezone(timezone.utc).isoformat('T', timespec)
        unix = iso_str.replace('+00:00', 'Z')
    return unix


def getDateFromUnix(unix):
    """
    This custom function accepts a Unix timestamp and converts it into the date format required by the Coinbase API
    """
    new_date = (datetime.utcfromtimestamp(unix).strftime('%Y-%m-%dT%H:%M:%S'))
    return new_date


def getNewEndDate(unix, interval):
    """
    This function will take a unix timestamp and subtract out 300 time intervals (in unix time) based on whether
    it is using daily, hourly, minute or 5min / 15min time intervals and return a new end date since Coinbase API
    will only process 300 candles max per request. Uses official date function to return date.
    """
    return getDateFromUnix(unix - (300 * int(interval)))


def OHLC_paginate(pair, starttime, endtime, interval):
    """Function to paginate through OHLC historical timeseries ...

    Accepts Coinbase Pair name using slash --> ie. 'BTC/USD'
    starttime = a timestamp in this format "%Y-%m-%d"  (see example)
    endtime = a timestamp in this format "%Y-%m-%d"  (see example)
    interval = granularity to pass. Options: "day", "hour", "minute", "5min", "15min"; default is daily

    After the function runs, it will continue adding candles until it grabs all of the available candles in the range
    passed to it.
    """

    new_end_time = starttime + 'T00:00:00'  # create timeformat out of the date passed for Coinbase

    # timestamps can get confusing; this code creates 2 unix, "epoch" timestamps out of the variables passed.
    start_unix = getUnixTimestamp(new_end_time)
    end_unix = getUnixTimestamp(endtime + 'T00:00:00')

    df_list = []  # set the empty list to store candlesticks until we capture them all
    # sym = pair.split('/')  # these two lines split the slash (/) out of the symbol 'BTC/USD' so will be BTC & USD
    # symbol = sym[0] + sym[1]
    # q_symbol = sym[0] + '-' + sym[1]  # symbol needs to be in format "BTC-USD" for Coinbase
    sym = pair.split('-')
    symbol = sym[0] + sym[1]
    q_symbol = sym[0] + '-' + sym[1]

    # set the unix time interval based on granularity passed
    if interval == 'day':
        time_period = '86400'
    elif interval == 'hour':
        time_period = '3600'
    elif interval == 'minute':
        time_period = '60'
    elif interval == '5min':
        time_period = '300'
    elif interval == '15min':
        time_period = '900'
    else:
        time_period = '86400'

    url = None  # initially set url variable to none to start looping from most recent data

    # We are going to loop until we reach a break condition
    while True:
        # if url = None, then script is just starting so start pulling latest data.
        if url is None:
            url = f"https://api.exchange.coinbase.com/products/{q_symbol}/candles?granularity={time_period}"  # format the URL
        else:
            url = f"https://api.exchange.coinbase.com/products/{q_symbol}/candles?granularity={time_period}&start={new_start_time}&end={new_end_time}"  # format the URL

        response = requests.get(url)  # get the request from Coinbase/ action the url param
        if response.status_code == 200:  # check if the response from Coinbase = 200 or "Good"
            data = json.loads(response.text)  # if its good, load the json response

            # these next few lines will take the json response and turn it into a pandas dataframe
            data_pd = pd.DataFrame(data, columns=['unix', 'low', 'high', 'open', 'close', 'volume'])
            data_pd['date'] = pd.to_datetime(data_pd['unix'],
                                             unit='s')  # Convert a human readable date out of unix timestamps
            data_pd['symbol'] = pair
            df_list.append(data_pd)  # append the dataframe candles to a list that we are storing them in
            print(
                f'Adding {len(data_pd["unix"])} candles with end time {getDateFromUnix(data_pd["unix"].iloc[-1])} for {symbol}')  # update user message on progress
            if len(data_pd["unix"]) == 0:  # check if len of json object is 0; if so, end loop
                break
            new_end_time = getDateFromUnix(
                data_pd['unix'].iloc[-1])  # reset the newest "startTime" to last date returned by previous API call
            new_start_time = getNewEndDate(unix=data_pd['unix'].iloc[-1],
                                           interval=time_period)  # get timestamp 300 periods back

            if data_pd['unix'].iloc[-1] < start_unix:  # we've gotten all of the data we need
                break

            if len(data_pd["unix"]) < 100:  # typically Coinbase returns 300 candles; if its less than 100,
                break  # then we are at the end and need to end our loop

        else:  # this else is for a server response other than "200 = Good"
            print(url)  # print url to screen of bad get request
            print(response.status_code)  # print status code and message and end loop
            print(response.text)
            break

    master_df = pd.concat(df_list)  # create a LARGER dataframe out of the list of dataframes we stored
    master_df = master_df.drop_duplicates(subset='unix', keep='first')  # drop any duplicate rows

    # sometimes we get more data in our dataframe that is outside of our current selected range so lets trim it
    new_master_df = master_df[master_df['unix'] > start_unix]
    new_master_df = new_master_df[new_master_df['unix'] < end_unix]

    return new_master_df  # return our dataframe with only the date ranges we want


if __name__ == "__main__":
    """
        Script Usage example Below to Pull Data from Coinbase for ANY time period and Symbol
    """

    start_date = (datetime.today() - timedelta(days=90)).isoformat()
    end_date = datetime.now().isoformat()

    START_DATE = '2013-01-01'
    END_DATE = '2022-06-08'
    GRANULARITY = 'day'
    objs = []
    for SYMBOL in names:
        # # the obj variable returned once the function runs is a pandas Dataframe
        obj = OHLC_paginate(pair=SYMBOL, starttime=START_DATE, endtime=END_DATE, interval=GRANULARITY)
        objs.append(obj)
        print(len(objs))
    output_df = pd.concat(objs)
    OUTPUT_FILENAME = f"Coinbase_{START_DATE.replace('-', '_')}_{END_DATE.replace('-', '_')}_{GRANULARITY}.csv"
    output_df.to_csv(OUTPUT_FILENAME, index=False)  # write the dataframe to CSV
