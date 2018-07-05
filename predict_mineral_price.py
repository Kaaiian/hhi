# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 13:22:39 2018

@author: Kaai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_HHI():
    file_name = r'HHI Indices by year, country, and element.xlsx'

    sheet_to_df_map = pd.read_excel(file_name, sheet_name=None)

    # set the years so the columns can be named as integers
    years = [1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007 ,2008,
             2009, 2010, 2011, 2012, 2013, 2014, 2015]

    df_HHI = sheet_to_df_map['Summary- HHI']
    df_HHI.drop(['Unnamed: 1'], inplace=True, axis=1)
    df_HHI.index = df_HHI['Element']
    df_HHI.drop(['Element'], inplace=True, axis=1)
    df_HHI.columns = years
    df_HHI.dropna(inplace=True, how='all')
    df_HHI.drop([np.NaN], inplace=True)
    return df_HHI


def get_prices():
    file_name = r'Yearly Commodity Price Estimator.xlsx'

    sheet_to_df_map = pd.read_excel(file_name, sheet_name=None)

    df_prices = sheet_to_df_map['Prices']
    df_prices.index = df_prices['Element']
    df_prices.drop(['Element'], inplace=True, axis=1)
    df_prices.dropna(inplace=True, how='all')
    df_prices = df_prices.iloc[:, 0:7]
    for index in list(set(df_prices.index)):
        df_prices.at[index, 'scaled_dollars'] = df_prices.loc[index, 'Dollars']/df_prices.loc[index, 'Dollars'].max()
        df_prices.at[index, 'Dollars (Max)'] = df_prices.loc[index, 'Dollars'].max()
    return df_prices


def get_oil_price():
    file_name = r'World Oil Production and Price.xlsx'
    df_oil = pd.read_excel(file_name)
    df_oil.index = df_oil['Year']
    df_oil = df_oil[['Production', 'Average Price', 'Price Index']]
    return df_oil


def entries_to_remove(entries, the_dict):
    for key in entries:
        if key in the_dict:
            del the_dict[key]

def get_total_production():
    file_name = r'HHI Indices by year, country, and element.xlsx'

    sheet_to_df_map = pd.read_excel(file_name, sheet_name=None)

    # set the years so the columns can be named as integers
    years = [1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007 ,2008,
             2009, 2010, 2011, 2012, 2013, 2014, 2015]

    entries = ('Notes by Element', 'Summary- HHI', 'Summary-Market Share (%)')
    entries_to_remove(entries, sheet_to_df_map)

#    index = pd.MultiIndex.from_tuples((), names=['year', 'element'])
    
    df_total_production = pd.DataFrame()
#    df_total_production.loc[('a', 'a'), 'test'] = 'kaai'

    for year in sheet_to_df_map:
        for element in sheet_to_df_map[year]['Element']:
            total_production = sheet_to_df_map[year]
            total_production.index = total_production['Element']
            value = total_production['Total World Production'].dropna()

            df_total_production[int(year.split()[0])] = value

    df_total_production = df_total_production.T
    df_scaled_total_production  = df_total_production/df_total_production.max()
    df_scaled_total_production = df_scaled_total_production.T
#    for column in df_scaled_total_production.columns.values:
#        plt.plot(df_scaled_total_production[column])
#        plt.show()
#        print(column)

    return df_scaled_total_production



#price_index = df_prices['Year','Price Index'] 
# %%


df_HHI = get_HHI()
df_prices = get_prices()
df_oil = get_oil_price().T
# total production is scaled by maximum values to elimante units
df_prod = get_total_production()


# %%


def get_y(df_prices):
    y = {}
    for index in list(set(df_prices.index)):
#        print(index)
        for year in df_prices.loc[index, 'Year']:
            if year == 1998:
                continue
            else:
                # year is indexed to match the feature matrix. (ie year 1999 = 1998 as index)
                y[(index, int(year)-1)] = df_prices[df_prices['Year'] == year].loc[index, 'scaled_dollars']
    return y


# %%


def get_max(df_prices):
    y = {}
    for index in list(set(df_prices.index)):
#        print(index)
        for year in df_prices.loc[index, 'Year']:
            if year == 1998:
                continue
            else:
                # year is indexed to match the feature matrix. (ie year 1999 = 1998 as index)
                y[(index, int(year)-1)] = df_prices[df_prices['Year'] == year].loc[index, 'Dollars (Max)']
    return y


# %%


def get_dollars(df_prices):
    y = {}
    for index in list(set(df_prices.index)):
#        print(index)
        for year in df_prices.loc[index, 'Year']:
            if year == 1998:
                continue
            else:
                # year is indexed to match the feature matrix. (ie year 1999 = 1998 as index)
                y[(index, int(year)-1)] = df_prices[df_prices['Year'] == year].loc[index, 'Dollars']
    return y


# %%

def get_X_price(df_prices):
    y = {}
    for index in list(set(df_prices.index)):
#        print(index)
        for year in df_prices.loc[index, 'Year']:
            if year == 2015:
                continue
            else:
                y[(index, int(year))] = df_prices[df_prices['Year'] == year].loc[index, 'scaled_dollars']
    return y


# %%


def get_X_HHI(df_HHI):
    y = {}
    for index in list(set(df_prices.index)):
#        print(index)
        for year in df_prices.loc[index, 'Year']:
            if year == 2015:
                continue
            else:
                y[(index, int(year))] = df_HHI.loc[index, year]
    return y


# %%


def get_X_prod(df_prod):
    y = {}
    for index in list(set(df_prices.index)):
#        print(index)
        for year in df_prices.loc[index, 'Year']:
            if year == 2015:
                continue
            else:
                y[(index, int(year))] = df_prod.loc[index, year]
    return y


# %%


def get_X_oil_avg(df_oil):
    y = {}
    for index in list(set(df_prices.index)):
#        print(index)
        for year in df_prices.loc[index, 'Year']:
            if year == 2015:
                continue
            else:
                y[(index, int(year))] = df_oil[year].loc['Average Price']
    return y


# %%


def get_X_oil_prod(df_oil):
    y = {}
    for index in list(set(df_prices.index)):
#        print(index)
        for year in df_prices.loc[index, 'Year']:
            if year == 2015:
                continue
            else:
                y[(index, int(year))] = df_oil[year].loc['Production']
    return y


# %%


def get_X_price_index(df_oil):
    y = {}
    for index in list(set(df_prices.index)):
#        print(index)
        for year in df_prices.loc[index, 'Year']:
            if year == 2015:
                continue
            else:
                y[(index, int(year))] = df_oil[year].loc['Price Index']
    return y


# %%
df_X = pd.DataFrame()

X_price_index = pd.Series(get_X_price_index(df_oil))
X_oil_prod = pd.Series(get_X_oil_prod(df_oil))
X_oil_avg = pd.Series(get_X_oil_avg(df_oil))
X_prod = pd.Series(get_X_prod(df_prod))
X_HHI = pd.Series(get_X_HHI(df_HHI))
X_price = pd.Series(get_X_price(df_prices))

df_X['X_price_index'] = X_price_index
df_X['X_oil_prod'] = X_oil_prod
df_X['X_oil_avg'] = X_oil_avg
df_X['X_prod'] = X_prod
df_X['X_HHI'] = X_HHI
df_X['X_price'] = X_price

y = pd.Series(get_y(df_prices))

dollars_max = pd.Series(get_max(df_prices))
dollars = pd.Series(get_dollars(df_prices))

# %%

df_formatted = df_X.copy()
df_formatted['Price (scaled, year+1)'] = y
df_formatted['Price'] = dollars
df_formatted['Price (Max)'] = dollars_max
df_formatted.dropna(inplace=True)

df_formatted.to_excel('formatted data.xlsx')























