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

def make_index():
    

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

    return df_HHI
#




#price_index = df_prices['Year','Price Index'] 


df_HHI = get_HHI()
df_prices = get_prices()
df_oil = get_oil_price()




x_i = df_HHI.loc[element, year]
y_i = df_prices.loc[element, year-1]



































