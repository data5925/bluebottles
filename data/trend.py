import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


def trend(target,beach):
    """
    target is a string for the target column whose trends we want to observe
    beach is an integer for the index of beach, ranging from 0,1,2
    Step1: Use the Dickey Fuller test to check for stationarity
    Step2: Using 4 different time intervals to check auto-correlation
    Step3: Decompose to visualize the trends
    """
    df = pd.read_csv('cleaned_data.csv')

    # extract the mean data of each month 
    df['time'] = pd.to_datetime(df['time'])
    df = df.groupby(['beach.x', df['time'].dt.strftime('%Y-%m')])[target].mean().reset_index()
    df = df[df['beach.x'] == beach]
    df = df[['time', target]]
    # set the index
    df.index = df['time']
    del df['time']

    # test stationarity
    """
    If the statistic is more negative than the tabulated critical value, at the 95% level, the null hypothesis of a unit root will be rejected.
    That means, the time serious data is stationary and will not have a trend or seasonality.
    """
    rolling_mean = df.rolling(3).mean()
    rolling_std = df.rolling(3).std()
    # visualize
    target_name = str(target).capitalize()
    plt.plot(df, color="blue", label="Original " + target_name + " data")
    plt.plot(rolling_mean, color="red", label="Rolling mean " + target_name)
    plt.plot(rolling_std, color="black", label="Rolling standard Deviation in " + target_name)
    plt.title(target_name + " Time Series, Rolling Mean, Standard Deviation")
    plt.legend(loc="best")
    plt.xticks(size = 7, rotation="vertical")
    plt.show()
    # augmented Dickey-Fuller test 
    adft = adfuller(df,autolag="AIC")
    output_df = pd.DataFrame({"Values":[adft[0],adft[1],adft[2],adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']], 
                          "Metric":["Test Statistics","p-value","No. of lags used","Number of observations used", "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
    print(output_df)

    # auto_correlation
    autocorrelation_lag1 = df[target].autocorr(lag=1)
    print("One Month Lag: ", autocorrelation_lag1)
    autocorrelation_lag3 = df[target].autocorr(lag=3)
    print("Three Month Lag: ", autocorrelation_lag3)
    autocorrelation_lag6 = df[target].autocorr(lag=6)
    print("Six Month Lag: ", autocorrelation_lag6)
    autocorrelation_lag9 = df[target].autocorr(lag=9)
    print("Nine Month Lag: ", autocorrelation_lag9)

    # decomposition
    decompose = seasonal_decompose(df[target],model='additive',period=7)
    decompose.plot()
    plt.xticks(size = 7, rotation="vertical")
    plt.show()

"""
result: I got a problem. Take trend('bluebottles', 0), trend('bluebottles', 1), trend('bluebottles', 2) as examples. 
Only the trend('bluebottles', 2) pass the test. And another problem is the choice for lag of auto-correlation. Here is the result:
 Values                       Metric
0  -1.322802              Test Statistics
1   0.618677                      p-value
2   7.000000             No. of lags used
3  26.000000  Number of observations used
4  -3.711212          critical value (1%)
5  -2.981247          critical value (5%)
6  -2.630095         critical value (10%)
One Month Lag:  0.10549810112382406
Three Month Lag:  -0.024481089264452898
Six Month Lag:  -0.3205959577982013
Nine Month Lag:  0.03661317360730444
"""
    
