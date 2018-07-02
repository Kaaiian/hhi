# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:37:15 2018

@author: Kaai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_HHI():
    file_name = r'HHI Indices by year, country, and element.xlsx'
    HHI_xlsx = pd.ExcelFile(file_name)

    HHI_xlsx.sheet_names

    sheet_to_df_map = pd.read_excel(file_name, sheet_name=None)

    years = [1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007 ,2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]
    
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
    HHI_xlsx = pd.ExcelFile(file_name)
    
    HHI_xlsx.sheet_names
    
    sheet_to_df_map = pd.read_excel(file_name, sheet_name=None)
    
    years = [1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007 ,2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]
    
    df_prices = sheet_to_df_map['Prices']
    df_prices.index = df_prices['Element']
    df_prices.drop(['Element'], inplace=True, axis=1)
    df_prices.dropna(inplace=True, how='all')

    return df_prices

df_HHI = get_HHI()
df_prices = get_prices()

# %%
years = np.arange(1998, 2016, 2)
for element in df_HHI.index:
    try:
        print(element)
        data_price = df_prices.loc[element, ['Year', 'Dollars']]
        data_price.index = df_prices.loc[element, 'Year']
        data_price.drop(['Year'], inplace=True, axis=1)
        data_hhi = df_HHI.loc[element]



        fig, ax1 = plt.subplots(figsize=(10, 10))
        font = {'family': 'DejaVu Sans',
                'weight': 'normal',
                'size': 18}
        plt.rc('font', **font)

        color_HHI = '#118ab2'
        ax1.plot(data_hhi, color=color_HHI, marker='o',
                     linestyle='-', mew=2, markerfacecolor='w', markersize=12)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('HHI', color=color_HHI)
        ax1.tick_params('y', colors=color_HHI)
        plt.xticks(years)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 2))

        ax2 = ax1.twinx()

        color_prices = '#176117'
        ax2.plot(data_price, color=color_prices, marker='o',
                     linestyle='-', mew=2, markerfacecolor='w', markersize=12)
        ax2.set_ylabel('Price ($)', color=color_prices)
        ax2.tick_params('y', colors=color_prices)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 2))
        plt.title(element)



        fig.savefig('figures/'+element+'_HHI_and_Price_vs_year.png')
        
        plt.show()

    except:
        pass

