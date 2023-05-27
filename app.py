import streamlit as st

# Create the sidebar menu using with st.columns()
menu_options = ['Graphs', 'Data', 'Comparison', 'Stock News']
menu_selection = st.sidebar.columns(1)[0].selectbox("Stock Dashboard", menu_options)

# Use the with statement to create tabs
with st.container():
    if menu_selection == 'Graphs':
        # Insert the main content of the web application
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import plotly.figure_factory as ff
        from pandas_datareader import data as pdr
        import yfinance as yf
        yf.pdr_override()
        from datetime import datetime, timedelta
        from keras.models import load_model

        # Set the start and end dates for the data
        startdate = datetime(2010, 1, 1)
        enddate = datetime(2019, 12, 31)

        st.title('Stock Price Prediction')

        # fetch the stock prices using pandas_datareader
        user_input = st.text_input('Enter Stock Ticker')
        if user_input != None:
            try:
                data = pdr.get_data_yahoo(user_input, start=startdate, end=enddate)
                if data.empty:
                    st.warning('No data found for the entered stock ticker. Please try again with a valid ticker.')
                else:
                    st.text(" ")
                    st.subheader('Graphs')
                    st.text(" ")

                    # Visualizations
                    st.subheader('Closing Price vs Time chart')
                    fig = plt.figure(figsize=(12, 6))
                    st.line_chart(data.Close)

                    st.subheader('Closing Price vs Time chart with 100MA')
                    ma100 = data.Close.rolling(100).mean()
                    df = pd.concat([data.Close, ma100], axis=1)
                    df.columns = ['Close', 'ma100']
                    st.line_chart(df)

                    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
                    ma100 = data.Close.rolling(100).mean()
                    ma200 = data.Close.rolling(200).mean()
                    fig = plt.figure(figsize=(12, 6))
                    df1 = pd.concat([data.Close, ma100, ma200], axis=1)
                    plt.plot(ma100)
                    df1.columns = ['Close', 'ma100', 'ma200']
                    st.line_chart(df1)

                    # Split data into training and testing

                    data_training = pd.DataFrame(data['Close'][0:int(len(data) * 0.70)])
                    data_testing = pd.DataFrame(data['Close'][int(len(data) * 0.70): int(len(data))])

                    # Normalize the data using MinMaxScaler
                    from sklearn.preprocessing import MinMaxScaler

                    scaler = MinMaxScaler(feature_range=(0, 1))

                    data_training_array = scaler.fit_transform(data_training)

                    # Load my model
                    model = load_model('keras_model.h5')

                    # Testing Part
                    past_100_days = data_training.tail(100)
                    final_data = past_100_days.append(data_testing, ignore_index=True)
                    input_data = scaler.fit_transform(final_data)

                    x_test = []
                    y_test = []

                    for i in range(100, input_data.shape[0]):
                        x_test.append(input_data[i - 100: i])
                        y_test.append(input_data[i, 0])

                    x_test, y_test = np.array(x_test), np.array(y_test)
                    y_predicted = model.predict(x_test)
                    scaler = scaler.scale_

                    scale_factor = 1 / scaler[0]
                    y_predicted = y_predicted * scale_factor
                    y_test = y_test * scale_factor

                    # Final Graph

                    st.subheader('Predicitons vs Original')
                    # fig2 = plt.figure(figsize=(12,6))
                    combined = np.concatenate([y_test.reshape(-1, 1), y_predicted.reshape(-1, 1)], axis=1)
                    df3 = pd.DataFrame(combined, columns=['Original Price', 'Predicted Price'])
                    st.line_chart(df3)

            except Exception as e:
                st.text("")
                
            finally:
                #Create a link to return to portfolio website
                st.subheader("Back to portfolio website")
                st.write("Click the link to return to portfolio website.")

                link = '[Portfolio](https://abhishekdj486.github.io/Portfolio/)'
                st.markdown(link, unsafe_allow_html=True)

    elif menu_selection == 'Data':
        import numpy as np
        import pandas as pd
        from pandas_datareader import data as pdr
        from datetime import datetime
        from keras.models import load_model

        startdate = datetime(2010, 1, 1)
        enddate = datetime(2019, 12, 31)

        st.title('Stock Price Prediction')

        user_input = st.text_input('Enter Stock Ticker')
        if user_input != None:
            try:
                data = pdr.get_data_yahoo(user_input, start=startdate, end=enddate)
                if data.empty:
                    st.warning('No data found for the entered stock ticker. Please try again with a valid ticker.')
                else:
                    # Describing data
                    st.subheader('Data from 2010-2019')
                    st.write(data.describe())

                    data_training = pd.DataFrame(data['Close'][0:int(len(data) * 0.70)])
                    data_testing = pd.DataFrame(data['Close'][int(len(data) * 0.70): int(len(data))])
                    # creating two columns
                    st.info('Same for all DataSets')
                    col1, col2 = st.columns(2)

                    # Display training data
                    with col1:
                        st.subheader('Training Data (70%)')
                        st.write(data_training)

                    # Display testing data
                    with col2:
                        st.subheader('Testing Data (30%)')
                        st.write(data_testing)


            except Exception as e:
                st.text("")

    elif menu_selection == 'Comparison':
        import matplotlib.pyplot as plt
        import yfinance as yf
        from datetime import datetime, timedelta
        from pandas_datareader import data as pdr

        st.title('Stock Price Prediction')

        # set stock symbol
        st.subheader('Comparison between two specific dates')
        user_input = st.text_input('Enter Stock Ticker')

        # Set the minimum and maximum date
        min_date = datetime.now() - timedelta(days=18250)
        max_date = datetime.now() 
        # set start and end dates
        startdate = st.date_input("Enter start date (YYYY-MM-DD): ", min_value=min_date, max_value=max_date)
        enddate = st.date_input("Enter end date (YYYY-MM-DD): ", min_value=min_date, max_value=max_date)


        # Fetching the stock price of date entered by user
        def get_data(user_input, startdate, enddate):
            # Fetch data from Yahoo Finance
            df = yf.download(user_input, start=startdate, end=enddate)  # interval='1d')
            # Extract the closing prices for the selected days
            prices = df['Close']
            # Return the closing prices as a list
            return prices.tolist()


        if user_input and startdate and enddate:
            try:
                prices = get_data(user_input, startdate, enddate)
                st.info(f"The Closing Price of {user_input} on {startdate} was : {prices[0]}")
                st.info(f"The Closing Price of {user_input} on {enddate} is : {prices[-1]}")

                # Display the difference textually
                if prices[0] > prices[-1]:
                    st.info(f'{startdate} has a high stock price when compared to {enddate}')
                elif prices[0] < prices[-1]:
                    st.info(f'{startdate} has a less stock price when compared to {enddate}')
                elif prices[0] == prices[-1]:
                    st.info(f'{startdate} and {enddate} have the equal stock price')

            except:
                st.warning("Invalid Ticker Symbol or Date Range")

        # Fetch stock prices
        if user_input != None:
            try:
                data = pdr.get_data_yahoo(user_input, startdate, enddate)
                if data.empty:
                    st.warning('No data found for the entered stock ticker. Please try again with a valid ticker.')
                else:
                    st.set_option('deprecation.showPyplotGlobalUse', False)

                    # plot closing prices
                    st.line_chart(data.Close)

            except Exception as e:
                st.text("")

    elif menu_selection == 'Stock News':
        import requests
        from bs4 import BeautifulSoup

        try:

            # Sraping the latest stock news from Yahoo Finance
            url = 'https://finance.yahoo.com/'
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_headlines = soup.find_all('h3', {'class': 'Mb(5px)'})

            # Displaying the news headlines using Streamlit
            st.write("## Top Stock News")
            for i, headline in enumerate(news_headlines):
                st.write(f"{i + 1}. {headline.text}")
        except Exception as e:
            st.info("Data cannot be fetched, Check your network connection")
