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
  - Incorporate the tones of online news articles into a traditional stock price forecasting model carried out with ubiquitous quantitative data (Open, Close, Adj-Close, Volume, High, Low) to improve the accuracy of adjusted-closing price forecast. In order for including the sentiment of articles, a binary text classification (sentiment analysis) is implemented on web news headlines with positive/negative labels.

* **Success Criteria:**
  - The forecasting model trained on both ticker and news sentiment data is expected to outperform the model trained only on the ticker data (Baseline). The used evaluation metrics are Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE). an A/B testing based on the performance measured by the two metrics can present the intervention of the sentiment data in a forecast and the effectiveness of the target model in comparison with the baseline model.
  
* **Work Flow:**
  1. Fine-tuning the pretrained DistilBert-base-uncased model on finance news headline dataset available in Kaggle
  2. Crawling and scraping web news headlines and the corresponding keywords (symbols)
  3. Selecting a single stock item whose symbol appears most frequently and gathering the corresponding headlines  
  4. Implementing text classification on the scraped headlines with postive and negative lables
  5. Scraping numeric ticker data of a stock item of interest and NASDAQ Composite
  6. Merging the classified headline data and the scraped ticker data
  7. Training a stacked LSTM RNN model on the preprocessed data
  8. Comparing the performance with the model trained only on the ticker data

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

## 2. Models

### DistilBert-Base-Uncased Model Fine-tuned on Kaggle News Data
  - **Usage:** Sentiment Analysis
  - **Hyperparameters:** 
    - *Loss Function:* SparseCategoricalCrossentropy
    - *Optimizer:* Adam
    - *Learning Rate:* Learning Rate Schedular
      - Start Rate = 0.01
      - End Rate = 1e-5
      - Power = 0.5
      - Decay Steps = len(train_x)/batch_size * num_epochs
    - *Epochs:* 20
    - *Batch Size:* 8
    - *Metric: Sparse Categorical Accuracy*
  
  - **Performance on Test Set:**
   
     **Accuracy** | **F1-Score**
     ------------|------------
     0.937 | 0.953


       


