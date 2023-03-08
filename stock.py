import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
from sklearn.metrics import mean_squared_error, r2_score



header = st.container()
dataset = st.container()
features = st.container()
Data_Development = st.container()
model_training = st.container()
your_time = st.container()
prediction = st.container()
copywright = st.container()

with header:
	st.title('Welcome to the BITCOIN (BTC_USD) prediction app')
	st.write('In this model, I developed a stock prediction system that can predict the price of a stock. I particularly focused on the BTC-USD. I scrapped off the closing price of BTC for each day from its invention to the 3rd March, 2023. With this data at hand, I trained a system to use this data and predict closing prices for this particular stock.')


	with dataset:
		st.header('BTC-USD Prices scrapped from yFinance')
		st.write('Here, the dataset contains about 3089 rows of Opening, Closing price, also the highest and the lowest price for each day, scrapped from yahoo finance website using python, with this data, I was able to train our model, exploring the dataset, we have no null cell, with the Opening, Closing, Lowest price, Highest price being float variable while the Volume is an integer.')

		data = pd.read_csv('./image/btc.csv')
		st.write(data.head())
		#X = data.drop('Close', axis = 1)
		#y = data['Close'] #close as our target variable.

		st.subheader('BTC_USD PRICE PREDICTOR')
		st.write('To visualize how our closing prices was distributed over the years')
		label_dist = pd.DataFrame(data['Close'])
		st.bar_chart(label_dist)

	with Data_Development:
		st.header('Data Development')
		st.write('In this section, I used the sklearn module to split our data into the training text and testing texts, we split in 70%-30%, and I further went ahead to use the LinearRegression algorithm for our model.')



	with model_training:
		st.header('Time to deploy the model!')
		st.write('Here, the model has been trained and saved using the linear regression algorithm. I also tested the accuracy of our model, it came back with an r2_score of 99% which is a good one for the model. The model was saved and is being used here for the deployment and prediction.')

		
		#lr = LinearRegression()
		#lr.fit(X_train, y_train)
		#y_pred = lr.predict(X_test)


		with your_time:
			st.header('Enter the necessary values here and predict!')
			#sel_col, disp_col = st.columns(2)

			Open = st.number_input('Open: Opening Price', step = 100)
			Low = st.number_input('Low: Lowest Price for the day', step = 100)
			High = st.number_input('High: Highest Price for the day', step = 100)
			Volume = st.number_input('Volume: No. of transactions', step = 100)

		def predict(data):
			lr = joblib.load('lr_model.sav')
			return lr.predict(data)
	
		with prediction:
			st.header('Click below to predict the closing price!')
			if st.button('Predict BTC Closing Price'):
				Close = predict(np.array([[Open, Low, High, Volume]]))
				st.write('The close price is', (Close[0]))








			with copywright:
				st.text('By Oluwaseyi Michael')