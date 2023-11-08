import pandas as pd
from application import app
from application import db
df = pd.read_csv('stocks.csv')
df.iloc[4538, 0] = 'NANO'
df.drop('% Change', axis = 1)
df['Market Cap'].fillna(0,inplace=True)
df['Country'].fillna('Global',inplace=True)
df['IPO Year'].fillna(0,inplace=True)
df['Sector'].fillna('Multi',inplace=True)
df['Industry'].fillna('Multi',inplace=True)
db.drop_all()
db.create_all()
from application.database import User, Wallet, Stock
for index, row in df.iterrows():
    stock = Stock(symbol=row['Symbol'], name=row['Name'], last_sale=row['Last Sale'],
                  net_change=row['Net Change'], market_cap=row['Market Cap'],
                  ipo_year=row['IPO Year'], volume=row['Volume'], country=row['Country'],
                  sector=row['Sector'], industry=row['Industry'])
    db.session.add(stock)
    print('sucess')

db.session.commit()
