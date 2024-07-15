
import numpy as np 
import pandas as pd    
class Car():
    
    def __init__(self, price):
        self.data = self.data = pd.DataFrame(price)
    def fitness_level(self, chromosome):
        # Convert the chromosome to two integers
        sma = int(''.join(map(str, chromosome[:6])), 2) + 1  # 1 <= length of window for shorter moving average <= 2^(n/2) - 1 
        lma = int(''.join(map(str, chromosome[6:])), 2) + sma+1 # sma < length of window for longer moving average <= sma + (2^(n/2) - 1)

        self.data["SMA"] = self.data["Close"].rolling(window= sma).mean()
        self.data["LMA"] = self.data["Close"].rolling(window= lma).mean()

        # Define the position
        '''
        • Go long (= +1) when the shorter SMA is above the longer SMA.
        • Go short (= -1) when the shorter SMA is below the longer SMA.
        For a long only strategy one would use +1 for a long position and 0 for a neutral position.
        '''
        self.data["Position"] = np.where(self.data['SMA'] > self.data['LMA'], 1, -1)

        self.data['Change'] = np.log(self.data["Close"] / self.data["Close"].shift(1))
        self.data['Return'] = self.data['Position'].shift(1)*self.data['Change']
        result  = self.data['Return'].sum()
        return result
    


raw = raw = pd.read_csv('./tr_eikon_eod_data.csv',
                              index_col=0, parse_dates=True)



symbol = 'AAPL.O'
data = pd.DataFrame(raw[symbol].dropna())
print(data)

data.rename(columns={'AAPL.O':'Close'}, inplace=True)

bb = Car(data)
fitness_lv = bb.fitness_level("000111001111")
print(fitness_lv)
print(bb.data)
