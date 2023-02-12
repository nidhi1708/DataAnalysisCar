import streamlit as st
from functools import cache

import sweetviz as sv
import requests



def data_graph(df , col):
    df = df.dropna(subset=[col]) 
    data_Graph = df[col].value_counts().reset_index().sort_values('index')
    data_Graph.rename(columns={'index': col, col: 'No of Models'}, inplace=True)
    return data_Graph


def clean_data(df):
    df = df.rename(columns={'Make': 'Company', 'Ex-Showroom_Price': 'Price'})
    df['City_Mileage']=df['City_Mileage'].apply(lambda x: str(x).replace('?' , '') if '?' in str(x) else str(x))
    df['City_Mileage']=df['City_Mileage'].apply(lambda x: str(x).replace(',' , '.') if ',' in str(x) else str(x))
    df['City_Mileage']=df['City_Mileage'].apply(lambda x: str(x).replace('km/litre' , '') if 'km/litre' in str(x) else str(x))
    df['City_Mileage']=df['City_Mileage'].apply(lambda x: str(x).replace('-12.7' , '') if '-12.7' in str(x) else str(x))
    df['City_Mileage']=df['City_Mileage'].apply(lambda x: str(x).replace('26032' , '26.03') if '26032' in str(x) else str(x))
    df['City_Mileage']=df['City_Mileage'].astype('float')

    df['Displacement']=df['Displacement'].apply(lambda x: str(x).replace('cc' , '') if 'cc' in str(x) else str(x))
    df['Displacement']=df['Displacement'].astype('float')

    df['Cylinders']=df['Cylinders'].fillna(4.0)

    df['Length']=df['Length'].apply(lambda x: str(x).replace('mm' , '') if 'mm' in str(x) else str(x))
    df['Length']=df['Length'].astype('float')

    df['Height']=df['Height'].apply(lambda x: str(x).replace('mm' , '') if 'mm' in str(x) else str(x))
    df['Height'].fillna('1387' ,inplace=True)
    df['Height']=df['Height'].astype('float')

    df['Width']=df['Width'].apply(lambda x: str(x).replace('mm' , '') if 'mm' in str(x) else str(x))
    df['Width'].fillna('1770' , inplace=True)
    df['Width']=df['Width'].astype('float')
    
    df['Fuel_Tank_Capacity']=df['Fuel_Tank_Capacity'].apply(lambda x: str(x).replace('litres' , '') if 'litres' in str(x) else str(x))
    df['Fuel_Tank_Capacity']=df['Fuel_Tank_Capacity'].astype('float')

    df['Kerb_Weight']=df['Kerb_Weight'].apply(lambda x: str(x).replace('kg' , '') if 'kg' in str(x) else str(x))
    df['Kerb_Weight']=df['Kerb_Weight'].apply(lambda x: str(x).replace('1016-1043 ' , '1030') if '1016-1043 ' in str(x) else str(x))
    df['Kerb_Weight']=df['Kerb_Weight'].apply(lambda x: str(x).replace('1053-1080 ' , '1067') if '1053-1080 ' in str(x) else str(x))
    df['Kerb_Weight'].fillna('1387.30' , inplace=True)
    df['Kerb_Weight']=df['Kerb_Weight'].astype('float')

    df['Rear_Brakes']=df['Rear_Brakes'].fillna('Drum')
    df['Front_Brakes']=df['Front_Brakes'].fillna('Ventilated Disc')

    return df


def temp_df(df):

    df = df.fillna('')
    df = df.replace(' ', '')

    df['Price'] = df['Price'].apply(lambda x: str(x).replace('Rs.', '') if 'Rs.' in str(x) else str(x))
    df['Price'] = df['Price'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else str(x))
    df['Price'] = df['Price'].apply(lambda x: int(x))
    return df



def comp_df(df):
    comp_df = df.dropna(subset=['Company']) #dropping all the columns which don't have a company
    comp_df = comp_df.fillna('')
    comp_df = comp_df.replace(' ', '')
    return comp_df

def top_cars(df , fuel_type):
    new_df = df.dropna(subset=['Company'])
    final_df = new_df[new_df['Fuel_Type'] == fuel_type]
    df['Price'] = df['Price'].str.split(' ').str.get(1).str.replace(',', '').astype('int')
    final_df_new = final_df[['Company', 'Model', 'Price', 'Fuel_Type']].sort_values(by=['Price'],
                        ascending=False).reset_index().drop(['index'], axis=1).head(10)

    return final_df_new


def get_comp_data(df , company):
    comp_df = df[df['Company'] == company]
    final_df_new = comp_df[['Company', 'Model', 'Price', 'Fuel_Type']].sort_values(by=['Price'],
                    ascending=False).reset_index().drop( ['index'], axis=1).head(10)
    return final_df_new


def sort_via_price(df , p1 , p2):
    df = df[df['Price'] >= p1]
    df = df[df['Price'] < p2]
    return df


def get_col_list(df , col):
    list_res = df[col].unique().tolist()
    list_res.sort()
    list_res.insert(0, '-')
    return list_res


def get_model_data(df , comp , model , price):
    temp = df[df['Company'] == comp]
    temp = temp[temp['Model'] == model]
    temp = temp[temp['Price'] == price]
    return temp


def generate_pandas_profile_report(df):
    from pandas_profiling import ProfileReport
    profile = ProfileReport(df)
    return profile  

def generate_sweetviz_report(df):
    report = sv.analyze(df)
    return report

#loading animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

        

def load_animaton(str):
    lottie_url_type = str
    lottie_type = load_lottieurl(lottie_url_type)
    return lottie_type
   