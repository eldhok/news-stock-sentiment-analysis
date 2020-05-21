from __future__ import (absolute_import, division, print_function,
                            unicode_literals)

import backtrader as bt
import backtrader.indicators as btind
import os.path
import sys
import nltk
import warnings
warnings.filterwarnings('ignore')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import pprint

from urllib.request import Request, urlopen

from pandas_datareader import data as pdr

#import fix_yahoo_finance as yf
import yfinance as yf

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io

import pyodbc

import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from io import BytesIO

#import MySQLdb

from pandas.io import sql

#import StringIO
import base64

import pandas as pd
from sqlalchemy import create_engine


nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# pip install mysql
# pip install mysqlclient
# pip install mysql-connector-python
import mysql.connector
from mysql.connector import errorcode


hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}
'''
hdr_ = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36','Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}
'''
def sql_download(n):
    connection_string = "Give Azure Connection String here"

    engn = create_engine(connection_string, echo=True)
    conn = engn.connect()
    #conn = mysql.connector.connect(**config)
    df = pd.read_sql("SELECT * FROM "+n+";",con = engn)
    print(df)
    conn.close()
    return df




def sql_upload(df,n):
    #print(df)
    connection_string = "Give Azure Connection String here"

    engn = create_engine(connection_string,echo=True)
    conn = engn.connect()
    df.to_sql(name = n, con = engn, index=False, if_exists='replace')
 
    conn.close()
    engn.dispose()

def plot_l(df,title):
    df = df.cumsum()

    df.plot(style='.-',title=title)


def plot_line(sales,sales1,sales2):
    

    n_groups = len(sales)
    

    # create plot
    index = np.arange(n_groups)
    line_chart1 = plt.plot(sales, sales1)
    line_chart2 = plt.plot(sales, sales2)
    
    plt.legend(['Stock Price', 'Sentiment'], loc=4)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Sentiment VS Stock Price')
    plt.legend()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.setp( ax.xaxis.get_majorticklabels(), rotation=50 ) 

    plt.show()


def plot_bar(coun_,fem_,male_):


    # To Plot Sex Rations in Each Country
    
    img = BytesIO()
    
    #clear plt
    #plt.clf() 
    #plt.cla()


    N = len(coun_)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27       # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    yvals = fem_
    rects1 = ax.bar(ind, yvals, width, color='r')
    zvals = male_
    rects2 = ax.bar(ind+width, zvals, width, color='g')

    ax.set_ylabel('Scores')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( coun_ )
    ax.legend( (rects1[0], rects2[0]), ('Famale', 'Male') )


    plt.setp( ax.xaxis.get_majorticklabels(), rotation=35 ) 

    for rect in ax.patches:
        # Get X and Y placement of label from rect.
       
       y_value = rect.get_height()
       x_value = rect.get_x() + rect.get_width() / 2
       if y_value !=0:

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        
        label = "{:}".format(float(y_value))

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.




    plt.savefig(img, format='png')
    img.seek(0)
    plt.imshow(img)
    plt.close(fig)


    return base64.b64encode(img.getvalue())


def plot_img(coun_,means_frank,means_guido):


    n_groups = len(coun_)
    

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, means_frank, bar_width,
    alpha=opacity,
    color='b',
    label='Stock')

    rects2 = plt.bar(index + bar_width, means_guido, bar_width,
    alpha=opacity,
    color='g',
    label='Sentiment')

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Sentiment VS Stock Price')
    plt.xticks(index + bar_width,coun_)
    plt.legend()


    plt.setp( ax.xaxis.get_majorticklabels(), rotation=50 ) 

    for rect in ax.patches:
        # Get X and Y placement of label from rect.
       
       y_value = rect.get_height()
       x_value = rect.get_x() + rect.get_width() / 2
       if y_value !=0:

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:}".format(float(round(y_value,2)))

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.




    plt.tight_layout()
    plt.show()

date_sentiments = {}

comp_name = input("\nEnter the Company name to Analyse Sentiment\n") 
stock_symb = input("\nEnter the Company Stock Symbol to get Stocks\n")


for i in range(1,10):
    
    #if i%2==0:
    #   req = Request('https://www.businesstimes.com.sg/search/facebook?page='+str(i), headers= hdr_)
    #else:
    req = Request('https://www.businesstimes.com.sg/search/'+comp_name+'?page='+str(i), headers= hdr)
    page = urlopen(req).read()
    #page = urlopen('https://www.businesstimes.com.sg/search/facebook?page='+str(i)).read()
    soup = BeautifulSoup(page, features="html.parser")
    posts = soup.findAll("div", {"class": "media-body"})
    for post in posts:
        time.sleep(1)
        url = post.a['href']
        date = post.time.text
        #print(date, url)
        try:
            req = Request(url, headers= hdr)
            link_page = urlopen(req).read()
            #link_page = urlopen(url).read()
        except:
            continue
            url = url[:-2]
            req = Request(url, headers= hdr)
            link_page = urlopen(req).read()
            #link_page = urlopen(url).read()
        link_soup = BeautifulSoup(link_page)
        sentences = link_soup.findAll("p")
        passage = ""
        for sentence in sentences:
            passage += sentence.text
        sentiment = sia.polarity_scores(passage)['compound']
        date_sentiments.setdefault(date, []).append(sentiment)

date_sentiment = {}

for k,v in date_sentiments.items():
    date_sentiment[datetime.strptime(k, '%d %b %Y').date() + timedelta(days=1)] = round(sum(v)/float(len(v)),3)

earliest_date = min(date_sentiment.keys())

last_date = max(date_sentiment.keys())

#print(date_sentiment)




yf.pdr_override() # <== that's all it takes :-)

# download dataframe
data = pdr.get_data_yahoo(stock_symb, start=earliest_date, end=last_date)




#print(data)
#print(date_sentiment)
Senti = (pd.DataFrame(list(date_sentiment.items())).rename(columns={0: "Date", 1: "Senti"}).set_index("Date"))
#print(Senti)


sql_upload(data.reset_index().fillna(0),"news_data")

news_data_download = sql_download("news_data").set_index("Date")



sql_upload(Senti.reset_index().fillna(0),"Senti_Analyis")

Senti_download  = sql_download("Senti_Analyis").set_index("Date")

#join_table = (Senti.join(data, how='inner')).reset_index().fillna(0)

# Use Downloaded Data

join_table = (Senti_download.join(news_data_download, how='inner')).reset_index().fillna(0)

join_table['Stock_Change'] = list(np.array(join_table["Close"].tolist()) - np.array(join_table["Open"].tolist()))

print(join_table)

# inner join
#join_table = (pd.merge(Senti, data, how='inner')).reset_index().fillna(0)
# left join
#join_table = ((Senti.join(data)).reset_index()).fillna(0)


#print(join_table)

#join_table = (Senti.join(data))
#print(datetime.strptime(join_table['Date'].values, '%d %b %Y').date())



#print(join_table.reset_index()) 
#print(join_table)
#li = (pd.to_datetime(join_table.index.values).tolist())
#print(list(np.array(pd.to_datetime(join_table['Date'].values, format='%Y-%m-%d').tolist())))
#li = list(np.array(pd.to_datetime(join_table['Date'].values, format='%Y-%m-%d')))
#li = pd.to_datetime(join_table['Date'].unique()).tolist()
#li = ((join_table.index).tolist())
#print(li.tolist())

#print(join_table["Senti"].values.tolist())
#print((pd.to_datetime(join_table.index,format='%d/%b/%Y')).values)
#print((join_table["Close"] - join_table["Open"]).values.tolist())


plot_l((join_table[['Date','Senti','Stock_Change']]).set_index('Date'),"Sentiment Score VS Stock Change 1")

#plot_l((join_table[['Date','Senti','Close']]).set_index('Date'),"Sentiment Score VS Stock Close")
#plot_line((join_table[['Date','Senti','Stock_Change']]).set_index('Date'),"Sentiment Score VS Stock Change 2")

plot_img(list(join_table["Date"].dt.date),join_table["Stock_Change"].tolist(),join_table["Senti"].tolist())

#plot_img(list(join_table["Date"].dt.date),list(np.array(join_table["Close"].tolist()) - np.array(join_table["Open"].tolist())),join_table["Senti"].tolist())

#plot_line(list(join_table["Date"].dt.date),list(np.array(join_table["Close"].tolist()) - np.array(join_table["Open"].tolist())),join_table["Senti"].tolist())

'''
stock_open = []
stock_close = []
stock_date = []
stock_senti = []

for key, value in date_sentiment.items(): 
    #print(key, ":", value)
    stock_senti.append(value)
    stock_date.append(key)

    #stock_open.append(data.loc[key])
    #print(data.loc[['Open']])

'''

# download Panel
#data = pdr.get_data_yahoo(["SPY", "IWM"], start="2017-01-01", end="2017-04-30")
