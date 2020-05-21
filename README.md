Sentimental Analysis also known as Emotional AI or Opinion mining is the use of language processing and analysis to study, extract and identify subjective information.

It is basically the measurement of positive and negative languages. 

The project uses news sentimental analysis to illustrate how stock prices and news articles are related.

- Sentimental Analysis is performed on news articles scrapped from news websites to generate corresponding Sentiment Scores
- Stock prices were collected from Yahoo finance
- Sentiment scores generated and Stock prices collected are uploaded to Azure Cloud
- Relationship between both are visualized through Graphs.

Technologies Used:

- nltk ‘vader lexicon’ : Perform  Sentimental analysis on Articles scrapped.
- BeautifulSoup : To scrape data from Business Times website.
- yfinance Python library : To retrieve stock data from Yahoo Finance.
- Azure Cloud : Store and retrieve Finance and Sentiment data as SQL tables.
- Matplotlib : Used for Visualisation in Python


