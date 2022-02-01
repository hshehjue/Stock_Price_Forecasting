# Stock Price Forecasting with LSTM-RNN and DistilBERT-base Deep Learning Models
### [Final Code](https://github.com/hshehjue/Stock_Price_Forecasting/blob/main/6.Final_Forecasting.ipynb) 

## Information 
* **Creator:** SeungHeon Han
* **Duration:** 1/20/2022 - 1/31/2022
* **Environment:**
  - Python v3.7.5 
  - Tensorflow v2.7.0
  - Keras v2.7.0
  - transformers(huggingface) v4.13.0
  - Google Colab Pro
    - GPU Info: 
      - NVIDIA-SMI 495.46
      - Driver Version: 460.32.03    
      - CUDA Version: 11.2

## 1. Executive Summary
* **Project Goal:**
  - Incorporate the tones of online news articles into a traditional stock price forecasting model carried out with ubiquitous quantitative data (Open, Close, Adj-Close, Volume, High, Low) to improve the accuracy of adjusted-closing price forecast. In order for including the sentiment of articles, a text classification (sentiment analysis) is implemented on web news headlines with positive/negative labels (neutral sentiment is included during a further step). The overall codes are written in a practical manner so that users can arbitrarily input any values of interest by using user-defined input() functions.

* **Success Criteria:**
  - The forecasting model trained on both ticker and news sentiment data is expected to outperform the model trained only on the ticker data (Baseline). The used evaluation metrics are Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE). an A/B testing based on the performance measured by the two metrics can present the contribution of the sentiment data to the forecast.
 
* **Work Flow:**
  1. Fine-tuning the pretrained DistilBert-base-uncased model on finance news headline dataset available in Kaggle
  2. Crawling and scraping web news headlines and the corresponding keywords (symbols)
  3. Selecting a single stock item whose symbol appears most frequently and gathering the corresponding headlines  
  4. Implementing text classification on the scraped headlines with postive and negative lables
  5. the same day usually has multiple articles. So, average the predicted sentiment values (1 or 0) by the number of articles on the same day, then:
  ```
  senti_by_time['sentiment'] = senti_by_time['sentiment'].apply(lambda x:"positive" if x > 0.5 
                                                                else ("negative" if x < 0.5 
                                                                else "neutral"))
  ```
  6. Scraping numeric ticker data of a stock item of interest and NASDAQ Composite
  7. Merging the classified headline data and the scraped ticker data
  8. Training a stacked LSTM RNN model on the preprocessed data
  9. Comparing the performance with the model trained only on the ticker data

* **Sources**
  * **NASDAQ Symbol List:**
    - [https://www.nasdaqtrader.com/](https://www.nasdaqtrader.com/)
  * **Kaggle News Data for Fine-tuning:**
    - [https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news/version/5?select=all-data.csv](https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news/version/5?select=all-data.csv)
  * **News Headlines for Forecasting:**
    - [https://www.benzinga.com/news](https://www.benzinga.com/news)
  * **Ticker Data:**
    - [Yahoo Finance Historical Data](https://finance.yahoo.com/)
  * **Pretrained Distilbert-base-uncased:**
    - Huggingface API

* **Training & Testing Description:**
  - Only about 3months-long news articles are accessible in Benzinga website
  - Scraping headlies related to **Apple(AAPL)** ranging from 11/2/2021 to 1/31/2022 
  - Scraping daily ticker data of **Apple(AAPL) & NASDAQ Composite(^IXIC)** ranging from 11/2/2021 to 1/31/2022 
  - Time Steps = 5 days
  - train : test = 80% : 20%
    - training with 49 days-long daily data
    - testing with 13 days-long daily data

## 2. Models

### I. DistilBert-Base-Uncased Model Fine-tuned on Kaggle News Data
  - **Usage:** Sentiment Analysis for News Headlines
  - **Hyperparameters:** 
    - *Loss Function:* Sparse Categorical Crossentropy
    - *Optimizer:* Adam
    - *Learning Rate:* Learning Rate Schedular
      - Start Rate = 0.01
      - End Rate = 1e-5
      - Power = 0.5
      - Decay Steps = len(train_x)/batch_size * num_epochs
    - *Epochs:* 20
    - *Batch Size:* 8
    - *Metric: Sparse Categorical Accuracy*
  - **Number of Parameters** 
  <img src=https://github.com/hshehjue/Stock_Price_Forecasting/blob/main/images/bert_parameters.png width=60% height=10%>
  
  - **Performance on Test Set:**
   
     **Accuracy** | **F1-Score**
     ------------|------------
     0.937 | 0.953

### II. Stacked-LSTM RNN
  - **Usage:** Time Series Forecast
  - **Target Stock Item:**
    - *Companies listed in NASDAQ Market*
    - ***Apple Inc (AAPL) in this case***
  <img src=https://github.com/hshehjue/Stock_Price_Forecasting/blob/main/images/apple_nasdaq.png width=800% height=20%>
  
  - **Variables:**
    - Target Feature (One-day-ahead): 
      - *Adjusted Closing Price*
    
    - Predictors (Time Steps = 5 days):
      - *Opening Price*
      - *Closing Price*
      - *Adj-Closing Price*
      - *High*
      - *Low*
      - *Closing Price of NASDAQ Composite (^IXIC)*
      - *Dummies for Positive/Negative/Neutral Sentiment*
    
  - **Hyperparameters:** 
    - *Time Steps:* 5 days
    - *Loss Function:* Mean Squared Error
    - *Optimizer:* Adam
    - *Epochs:* 5
    - *Batch Size:* 8
    - *Metric: Mean Squared Error*
    - *# of LSTM Layers:* 3
    - *Output Space:* 50
    - *Activation:* tanh
    - *Recurrent Activation:* sigmoid
```
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(y_train.shape[1]))
```

  - **Optimal Epochs = 5**
  <img src=https://github.com/hshehjue/Stock_Price_Forecasting/blob/main/images/lstm_process.png width=60% height=60%>
  
  - **Number of Parameters** 
  <img src=https://github.com/hshehjue/Stock_Price_Forecasting/blob/main/images/lstm_parameters.png width=60% height=10%>


## 3. Overall Performance 
 
  * **Baseline Model**
    - the identical stacked-LSTM RNN Time Series Forecasting model 
    - DOES NOT include the sentiment of news headlines as a predictor 

  * **Evaluation Metrics**
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Percentage Error (MAPE)
  
  * **Comparison**
  
    _| **RMSE** | **MAPE**
    ---|---|---
    **Baseline** | 5.99 | 0.19
    **Model** | 5.06 | 0.17 

<img src=https://github.com/hshehjue/Stock_Price_Forecasting/blob/main/images/performance.png width=80% height=80%>

* **Prediction Plot**
<img src=https://github.com/hshehjue/Stock_Price_Forecasting/blob/main/images/final.png width=80% height=60%>

## 4. Conclusion

  - the model with the sentiment variable has improved upon the baseline in RMSE and MAPE by 0.93 and 0.02 respectively. The model showed the increased accuracy over the additional implementations but at a distinct degree of the improvement. It implies that news articles can somewhat influence the movement of stock prices 
but this experiement cannot offer the high extent of its contributions due to the errors of sentiment analysis and the difficulty in dealing with neutral sentiment of news articles. So, I would conclude that the built model lacks the utility in leveraging for the actual investment activities.

## 5. Limitations & Potential Solutions
  1. Fine-tuning DistilBERT model on a labeled dataset available in Kaggle 
    - the Kaggle dataset also consists of web news headlines but not perfectly compatible with the input headline data collected from Benzinga.com 
    - the accuracy on test set does not guarantee the accuracy of the text classification of the headlines scraped from Benzinga.com
    - ***Potential Solution***
      - once a source of news is decided, the articles or headlines has to be manually labeled to train BERT 
        
  2. Benzinga website posts only about 3 months-long news articles
    - it bothers training a model with large historical data, which makes it difficult to drive up the overall prediction accuracy
    - ***Potential Solution***
      - Use another source where longer-term news articles are available 
      
  3. Sentiment analysis of news headlines cannot accurately predict the tones of the whole articles
    - scraping daily-updated whole ariticles is highly costly
    - ***Potential Solution***
      - find a source that offers news articles by company so that I can take only the articles of interest
  
  4. multiple articles about the same company are released on the same day
    - the multiple articles can deliever different tones so the daily sentiment data of a specific company cannot be always coherent 
    - ***Potential Solution***
      - incorporate neutral sentiment as well and take the same method I did in this experiment 
    
   
