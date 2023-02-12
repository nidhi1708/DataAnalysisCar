from matplotlib.style import use
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.figure_factory as ff
from functools import cache
import helper
from streamlit_lottie import st_lottie
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components
import pickle
import codecs
import requests

#Setting Up the Page confriguration
st.set_page_config(
    page_title="Data Analysis",
    layout="centered",
    initial_sidebar_state="auto",
)


#lottie animition urls for different user interfaces
body_type_animation='https://assets5.lottiefiles.com/packages/lf20_fdxsy2co.json'
price_type_animation='https://assets7.lottiefiles.com/packages/lf20_b4yychpi.json'
fuel_type_animation='https://assets8.lottiefiles.com/private_files/lf30_vdqxTM.json'
overall_animation_car='https://assets10.lottiefiles.com/packages/lf20_asjtnqce.json'
company_type_animation='https://assets4.lottiefiles.com/private_files/lf30_zcwz0fha.json'
predict_price_animation='https://assets1.lottiefiles.com/packages/lf20_3x67gx4y.json'
model_animation='https://assets1.lottiefiles.com/packages/lf20_5aaicf2r.json'
browse_data_animation='https://assets1.lottiefiles.com/packages/lf20_xmkgn4jj.json'

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

 
#Function to generate sweetviz within our website
def st_display_sweetviz(report_html , width=800,height=500):
	report_file = codecs.open(report_html,'r')
	page = report_file.read()
	components.html(page,width=width,height=height,scrolling=True)
 
 
def load_data():
    df = pd.read_csv('cars_engage_csv_3.csv')  # reading cars dataset
    final_df=helper.clean_data(df)
    return final_df


df=load_data()
temp_df = helper.temp_df(df)  # df after changes in Price column
comp_df = helper.comp_df(df)  # df after removing rows which don't have a company name


#Callback function for usermenu sidebar clicks
def handle_usermenu_sidebar_click():
    if st.session_state.selected_analysis:
        st.session_state.kind_of_analysis=st.session_state.selected_analysis  


#Setting up the sidebar
st.sidebar.title("Cars Data Analysis")
st.session_state['kind_of_analysis']=st.sidebar.radio(
    'Select an Option',(
         'Overall Analysis' , 'Price Wise Analysis' , 'Model-wise Comparision' , 'Body Type Wise Analysis' ,
        'Fuel Type Analysis' , 'Company-wise Analysis' , 'Predict Price' , 'Browse Data' 
    ) , on_change=handle_usermenu_sidebar_click , key='selected_analysis'
)

if st.session_state.kind_of_analysis=='Overall Analysis':
    company = df['Company'].unique().shape[0]
    models = df['Model'].unique().shape[0]

    st.title("Overall Analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.header("Companies: " + str(company))

    with c2:
        st.header("Models: " + str(models))
    
    # creating lottie animations
    st_lottie(helper.load_animaton(overall_animation_car), key="car  overall")

    st.header("Different Models made by the companies with there Body type")
    fig = px.scatter(temp_df, x="Company", y="Model", color="Body_Type", hover_data=['Price'])
    st.plotly_chart(fig)

    #Engine location in most of the cars
    fig, ax = plt.subplots(figsize=(16, 7))
    ax = sns.countplot(data=df, y='Engine_Location', alpha=.6, color='green')
    st.title('Cars by Engine_Location')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('')
    plt.ylabel('')
    st.pyplot(fig)

    #Correlation plot
    st.title("Correlation Plot")
    fig, ax = plt.subplots(figsize=(16, 16))
    ax = sns.heatmap(df.corr(),cmap='rocket_r',fmt=".2f", annot=True)
    st.pyplot(fig)
    st.write("From the heatmap we can see that The more the number of cylinders in a car, the more will be its displacement. Generally speaking, the higher an engine’s displacement the more power it can create. Similarly we can perform more analysis. Here unnamed:0 is by default which stands for number of models.")
     

    #Common analysis
    #line charts
    col=['Company' ,'Fuel_Type' ,'Body_Type' , 'Fuel_Tank_Capacity' , 'Cylinders' , 'Kerb_Weight' , 'Seating_Capacity']
    st.title("Common Analysis")
    for c in col:
        data_vs_model = helper.data_graph(df, c)
        fig = px.line(data_vs_model, x=c, y="No of Models")
        st.header(c +" Vs Number of Models")
        st.plotly_chart(fig)

    #pie charts
    fig = plt.figure(figsize=(10, 10))
    ax = fig.subplots()
    df.Cylinder_Configuration.value_counts().plot(ax=ax, kind='pie')
    ax.set_ylabel("")
    st.header("Cars Count by Cylinder_Configuration ")
    st.pyplot(fig)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.subplots()
    df.Emission_Norm.value_counts().plot(ax=ax, kind='pie')
    ax.set_ylabel("")
    st.header("Cars Count by Emission_Norm ")
    st.pyplot(fig)


#In Company Wise Analysis analysing different filds w.r.t Company
elif st.session_state.kind_of_analysis=='Company-wise Analysis':
    st.title("Company Wise Analysis")
 
    # creating lottie animations
    st_lottie(helper.load_animaton(company_type_animation), key="car company type")

    #User selecting required Companies for comparisions
    new_df = df.dropna(subset=['Company'])
    final_df = helper.temp_df(new_df)
    company_list = final_df['Company'].unique().tolist()
    company_selection = st.sidebar.multiselect('Company:', company_list, default=company_list)
    mask = final_df['Company'].isin(company_selection)

    #printing the top models of select companies
    st.header("Top Models of Selected Companies")
    group_comp_df=final_df[mask].groupby('Company')
    final_group_comp_df=group_comp_df[['Model' ,'Variant' ,'Price' , 'Fuel_Type' , 'Fuel_Tank_Capacity']]
    final_comp_df=final_group_comp_df.max()
    st.dataframe(final_comp_df)

    #Plotting Charts
    st.subheader('Relation between Price , Displacement and Fuel type of Various Companies')
    fig = px.scatter_3d(final_df[mask], x='Displacement', z='Price', y='Fuel_Type', color='Company')
    fig.update_layout(showlegend=True, autosize=True)
    st.plotly_chart(fig, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.scatterplot(data=final_df[mask], x='Company', y='Price', hue='Body_Type', palette='viridis', alpha=.89,
                         s=120)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Company', fontsize=14)
    plt.xticks(rotation=90)
    plt.ylabel('Price', fontsize=14)
    st.header('Relation between Price and Company')
    st.pyplot(fig)

    data_vs_model = helper.data_graph(final_df[mask], 'Company')
    fig = px.bar(data_vs_model, x='Company', y="No of Models", title="Number of Model per Company",
                 color_discrete_sequence=['#F63366'],
                 template='plotly_white')
    st.plotly_chart(fig)
   

    col_list = ['Price', 'Cylinders', 'Variant' , 'Seating_Capacity' , 'City_Mileage']
    for col in col_list:
        fig = px.scatter(final_df[mask], x='Company', y=col, title="Relationship between Company and " + col,
                         color='Fuel_Type')
        st.plotly_chart(fig)

    st.header('Relation between Width , length ,height and weigth of Cars of various Companies')
    fig = px.scatter(final_df[mask], x="Width", y="Length", color="Company", hover_data=['Height', 'Price'])
    st.plotly_chart(fig)

    st.header('Relation between Front & Rear Brakes of Cars of various Companies')
    fig = px.scatter(final_df[mask], x="Company", y=["Front_Brakes", "Rear_Brakes"])
    fig.update_layout(autosize=False, width=800, height=600)
    st.plotly_chart(fig)

#Comparing Models compares two models based on the given features 
elif st.session_state.kind_of_analysis=='Model-wise Comparision':
    st.title("Model-wise Comparision")

    c1, c2 = st.columns(2)
    with c1:
        company_list = np.unique(comp_df['Company'].dropna().values).tolist()
        company_list.sort()
        company_list.insert(0, '-')
        selected_comp = st.selectbox('Select a Company', company_list)

        new_df = comp_df[comp_df['Company'] == selected_comp]
        selected_model = st.selectbox('Select a Model', helper.get_col_list(new_df , 'Model'))

        final_df = new_df[new_df['Model'] == selected_model]
        selected_price = st.selectbox('Select the Price', helper.get_col_list(final_df , 'Price'))

    with c2:
        company_list_sec = np.unique(comp_df['Company'].dropna().values).tolist()
        company_list_sec.sort()
        company_list_sec.insert(0, '-')
        selected_comp_sec = st.selectbox('Select Company for 2nd Car', company_list_sec)

        new_df = comp_df[comp_df['Company'] == selected_comp_sec]
        selected_model_sec = st.selectbox('Select Model for 2nd Car', helper.get_col_list(new_df , 'Model'))

        final_df = new_df[new_df['Model'] == selected_model_sec]
        selected_price_sec = st.selectbox('Select Price for 2nd Car', helper.get_col_list(final_df , 'Price'))

    if (selected_comp != '-' and selected_model != '-' and selected_price != '-' and
        selected_comp_sec != '-' and selected_model_sec != '-' and selected_price_sec != '-'):

        #Extracting required row from dataset
        temp = helper.get_model_data(comp_df , selected_comp , selected_model , selected_price)
        temp_sec=helper.get_model_data(comp_df , selected_comp_sec , selected_model_sec , selected_price_sec)

        #Concating both the rows
        final_df=pd.concat([temp , temp_sec])
        final_df = final_df.astype(str)

        col = final_df.columns.tolist()
        col.pop(0)   

        #Showing the selected columns to end user
        st.subheader("Comparision")

        default_col_list=['Company' , 'Model']
        selected_columns = st.sidebar.multiselect("Search Features", col , default=default_col_list)
        new_df = final_df[selected_columns]
        res_df=new_df.reset_index().drop(['index'], axis=1)
        res_df_trans=res_df.T
        st.table(res_df_trans)

    else:
        # creating lottie animations
        st_lottie(helper.load_animaton(model_animation), key="car model type")

#Variation of Body Type w.r.t various fields
elif st.session_state.kind_of_analysis=='Body Type Wise Analysis':
    st.title("Body Type Wise Analysis ")
    #st.image("images\Bugatti_Chiron.jpg")

    # creating lottie animations
    st_lottie(helper.load_animaton(body_type_animation), key="car body type")

   #Asking the user to select Body_Type of cars , hence showing graphs according to that
    new_df = df.dropna(subset=['Body_Type', 'Company'])
    final_df = helper.temp_df(new_df)
    body_type_list = final_df['Body_Type'].unique().tolist()
    body_type_selection = st.sidebar.multiselect('Body Type:',body_type_list,default=body_type_list)
    mask = final_df['Body_Type'].isin(body_type_selection)

    st.header("Top Models of Selected Body Types")
    group_comp_df=final_df[mask].groupby('Body_Type')
    final_group_comp_df=group_comp_df[['Company', 'Model' ,'Variant' ,'Price' , 'Fuel_Type']]
    final_comp_df=final_group_comp_df.max()
    st.dataframe(final_comp_df)

    data_vs_model = helper.data_graph(final_df[mask], 'Body_Type')
    fig = px.bar(data_vs_model, x='Body_Type', y="No of Models" , title="Number of Model per Body Type", color_discrete_sequence=['#F63366'],
                         template='plotly_white')
    st.plotly_chart(fig)


    col_list = ['Price', 'Fuel_Type', 'Cylinders', 'Variant' , 'Seating_Capacity' , 'City_Mileage']
    for col in col_list:
        fig = px.box(final_df[mask], x='Body_Type', y=col,title="Relationship between Body Type and " +col ,color='Company')
        st.plotly_chart(fig)


    col_list=['Doors' ,'Model' , 'Company']
    for col in col_list:
        fig = px.scatter(final_df[mask], x="Body_Type", y=col,title="Body Type Vs "+col, color_discrete_sequence=['#F63366'],
                         template='plotly_white')
        st.plotly_chart(fig)


#Fuel_Type wise analysis analyses fuel_type changes depending on various feilds 
elif st.session_state.kind_of_analysis=='Fuel Type Analysis':
    st.title("Fuel Type Wise Analysis")
    #st.image("images\Ferrari-812.jpg")

    # creating lottie animations
    st_lottie(helper.load_animaton(fuel_type_animation), key="car fuel type")

    #Asking the user to select Fuel_Type of cars , hence showing graphs according to that
    new_df = df.dropna(subset=['Fuel_Type', 'Company'])
    final_df = helper.temp_df(new_df)
    fuel_type_list = final_df['Fuel_Type'].unique().tolist()
    fuel_type_selection = st.sidebar.multiselect('Fuel Type:',fuel_type_list,default=fuel_type_list)
    mask = final_df['Fuel_Type'].isin(fuel_type_selection)


    data_vs_model = helper.data_graph(final_df[mask], 'Fuel_Type')
    fig = px.line(data_vs_model, x='Fuel_Type', y="No of Models" , title="Number of Model per Fuel Type", color_discrete_sequence=['#F63366'],
                         template='plotly_white')
    st.plotly_chart(fig)

    col_list = ['Price', 'Cylinders', 'Variant' , 'Seating_Capacity' , 'City_Mileage']
    for col in col_list:
        fig = px.scatter(final_df[mask], x='Fuel_Type', y=col,title="Relationship between Fuel Type and " +col ,color='Company')
        st.plotly_chart(fig)


    col_list=['Model' , 'Company']
    for col in col_list:
        fig = px.scatter(final_df[mask], x="Fuel_Type", y=col,title="Fuel Type Vs "+col, color_discrete_sequence=['#F63366'],
                         template='plotly_white')
        st.plotly_chart(fig)


#Price wise analysis analyses prices of the cars depending on various feilds 
elif st.session_state.kind_of_analysis=='Price Wise Analysis':
    st.title("Price Wise Analysis")
    
    # creating lottie animations
    st_lottie(helper.load_animaton(price_type_animation), key="car price type")

    new_df = df.dropna(subset=['Price', 'Company'])
    final_df = helper.temp_df(new_df)

    #Allowing the user to select a budget and according to that we showing the plots , initially the budget is 0-max price
    price_list = ['Overall','1-5 Lakh' , '5-10 lakh' , '10-15 Lakh' , '15-20 Lakh' , '20-35 Lakh' , '35-50 Lakh' , 'Luxury Cars']
    selected_price = st.sidebar.selectbox('Select Your Budget',price_list)
    
    #Getting the required Dataset based on given budget
    if selected_price==price_list[1]:
        final_df=helper.sort_via_price(final_df , 0 , 500000)
    if selected_price==price_list[2]:
        final_df=helper.sort_via_price(final_df , 500000 , 1000000)
    if selected_price == price_list[3]:
        final_df = helper.sort_via_price(final_df, 1000000, 1500000)
    if selected_price==price_list[4]:
        final_df = helper.sort_via_price(final_df, 1500000, 2000000)
    if selected_price==price_list[5]:
        final_df = helper.sort_via_price(final_df, 2000000, 3500000)
    if selected_price==price_list[6]:
        final_df = helper.sort_via_price(final_df, 3500000, 5000000)
    if selected_price==price_list[7]:
        final_df=final_df[final_df['Price']>5000000]

     #Plotting   

    selected_sort=st.selectbox('Sort by', ['High to Low' , 'Low  to High'])
    order=False
    if selected_sort=='Low  to High':
        order=True
    final_df_new = final_df[['Company', 'Model', 'Price', 'Fuel_Type']].sort_values(by=['Price'],
                                                                                   ascending=order).reset_index().drop(
        ['index'], axis=1)
    st.dataframe(final_df_new)


    col_list = ['Cylinders', 'Variant' , 'Seating_Capacity' , 'City_Mileage']
    for col in col_list:
        fig = px.scatter(final_df, x='Price', y=col,title="Relationship between Price and " +col ,color='Company')
        st.plotly_chart(fig)


    col_list=['Model' , 'Company']
    for col in col_list:
        fig = px.scatter(final_df, x="Price", y=col,title="Price Vs "+col, color_discrete_sequence=['#F63366'],
                         template='plotly_white')
        st.plotly_chart(fig)


#Predicting Price Using RandomForest as it is most accurate one 
elif st.session_state.kind_of_analysis=='Predict Price':

    st.title("Selling Price Predictor")

    # creating lottie animations
    st_lottie(helper.load_animaton(predict_price_animation), key="car price predict")

    st.subheader("Find the selling price for your Car:")

    model = pickle.load(open('RF_price_predicting_model.pkl','rb'))

    #Taking User Details like year , present price , fuel type etc and predicting the price accordingly
    years = st.number_input('In which year car was purchased ?',1990, 2021, step=1, key ='year')
    Years_old = 2021-years

    Present_Price = st.number_input('What is the current ex-showroom price of the car ?  (ex: 1.00 = 1 Lakh)', 0.00, 50.00, step=0.5, key ='present_price')

    Kms_Driven = st.number_input('What is distance completed by the car in Kilometers ?', 0.00, 500000.00, step=500.00, key ='drived')

    Owner = st.radio("The number of owners the car had previously ?", (0, 1, 3), key='owner')

    Fuel_Type_Petrol = st.selectbox('What is the fuel type of the car ?',('Petrol','Diesel', 'CNG'), key='fuel')
    if(Fuel_Type_Petrol=='Petrol'):
        Fuel_Type_Petrol=1
        Fuel_Type_Diesel=0
    elif(Fuel_Type_Petrol=='Diesel'):
        Fuel_Type_Petrol=0
        Fuel_Type_Diesel=1
    else:
        Fuel_Type_Petrol=0
        Fuel_Type_Diesel=0

    Seller_Type_Individual = st.selectbox('Are you a dealer or an individual ?', ('Dealer','Individual'), key='dealer')
    if(Seller_Type_Individual=='Individual'):
        Seller_Type_Individual=1
    else:
        Seller_Type_Individual=0	

    Transmission_Mannual = st.selectbox('What is the Transmission Type ?', ('Manual','Automatic'), key='manual')
    if(Transmission_Mannual=='Mannual'):
        Transmission_Mannual=1
    else:
        Transmission_Mannual=0

    #Predicting selling price according to the filled data
    if st.button("Estimate Price", key='predict'):
        try:
            Model = model  #get_model()
            prediction = Model.predict([[Present_Price, Kms_Driven, Owner, Years_old, Fuel_Type_Diesel, Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Mannual]])
            output = round(prediction[0],2)
            if output<0:
                st.warning("You will be not able to sell this car !!")
            else:
                st.success("You can sell the car for {} lakhs 🙌".format(output))
        except:
            st.warning("Opps!! Something went wrong\nTry again")


else:
    st.title("Data Analysis") #Doing EDA on user provided dataset

    data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
    if data is None:
        # creating lottie animations
        st_lottie(helper.load_animaton(browse_data_animation), key="browse data animation")

    if data is not None:
        activities = ["EDA","Pandas Profiling Report" , 'Sweetviz Report']
        choice = st.sidebar.selectbox("Select Activities", activities)
        df = pd.read_csv(data)

        # In EDA we are going to show some general analysis and some plots 
        # These plots are going to visualize the Dataset more effectively and Quickly 
        # So I have included some visuals here like Animated Plots , 3D , 2D , Heatmaps etc

        if choice == 'EDA':
            st.subheader("Exploratory Data Analysis")
            col=df.columns.to_list()
            
            st.dataframe(df.head())

            analyse=["Basic Features","2D , Animated & Facet Plots" , "Pie Charts" , "Heatmaps, 3D & Ternary Plots"]
            feature=st.sidebar.selectbox("Choose an Option" , analyse)

            if feature=="Basic Features":
                if st.checkbox("Show Shape"):
                    st.write(df.shape)

                if st.checkbox("Show Columns"):
                    all_columns = df.columns.to_list()
                    st.write(all_columns)

                if st.checkbox("Summary"):
                    try:
                        st.write(df.describe())
                    except:
                        st.warning("df.describe() cannot be performed on the given Dataset")

                if st.checkbox("Show Selected Columns"):
                    all_columns = df.columns.to_list()
                    selected_columns = st.multiselect("Select Columns", all_columns)
                    new_df = df[selected_columns]
                    st.dataframe(new_df)

                if st.checkbox("Correlation Plot(Seaborn)"):
                    try:
                       fig, ax = plt.subplots(figsize=(16, 7))
                       ax=sns.heatmap(df.corr(),cmap='rocket_r',fmt=".1f", annot=True)
                       st.pyplot(fig)
                    except:
                        st.warning('Correlation Plot cannot be performed on the given Dataset')

                if st.checkbox("Value Count Plot"):
                    try:
                        selected_col = st.selectbox("Choose a col",col)
                        df = df.dropna(subset=[selected_col]) 
                        data_Graph = df[selected_col].value_counts().reset_index().sort_values('index')
                        data_Graph.rename(columns={'index': selected_col, selected_col: 'Counts'}, inplace=True)
                        fig = px.bar(data_Graph, x=selected_col, y='Counts', color_discrete_sequence=['#F63366'],template='plotly_white')
                        st.plotly_chart(fig)
                    except:
                        st.warning('Rows & Columns of the given dataset is not accurate for this plot.')    


            if feature=="2D , Animated & Facet Plots":
                st.header("Basic Charts")
                x_axis = st.selectbox("Choose x axis",col)
                y_axis = st.selectbox("Choose y axis",col)
                color_col=st.selectbox("Choose color col",col)
                hover_cols=st.multiselect("Do you want to hover any data" , col)
                plot_type=st.selectbox("Choose the plot type" , ['Scatter' , 'Line' , 'Bar' , 'Box' , 'Violin'])
                final_df = df.dropna(subset=[color_col])
                if st.button('Generate 2D Plot'):
                    if plot_type=='Scatter':
                        fig=px.scatter(final_df , x=x_axis , y=y_axis , color=color_col , hover_data=hover_cols)
                    if plot_type=='Line':
                        fig=px.line(final_df , x=x_axis , y=y_axis , color=color_col , hover_data=hover_cols)
                    if plot_type=='Box':
                        fig=px.box(final_df , x=x_axis , y=y_axis , color=color_col , hover_data=hover_cols)  
                    if plot_type=='Bar':
                        fig=px.bar(final_df , x=x_axis , y=y_axis , color=color_col , hover_data=hover_cols) 
                    if plot_type=='Violin':
                        fig=px.violin(final_df , x=x_axis , y=y_axis , color=color_col , hover_data=hover_cols)   
                    st.plotly_chart(fig)        

                if st.checkbox('Generate Animated Plots'):
                    animated_frame = st.selectbox("Choose a column for Animation", col)
                    plot_type_animated=st.selectbox("Choose the plot type" , ['Scatter' , 'Bar'])
                    new_df = final_df.dropna(subset=[animated_frame])
                    if st.button("Show Plot"):
                        if plot_type_animated=='Scatter':
                            fig=px.scatter(new_df, x=x_axis, y=y_axis, animation_frame=animated_frame, 
                               color=color_col, log_x=True)
                        if plot_type_animated=='Bar':
                            fig=px.bar(new_df, x=x_axis, y=y_axis, animation_frame=animated_frame, 
                               color=color_col, log_x=True)
                        st.plotly_chart(fig)


                if st.checkbox('Facet Plot'):
                    facet_row_col=st.selectbox('Choose Facet row' , col)
                    facet_column_col=st.selectbox('Choose Facet column' , col)    
             
                    if st.button('Plot Facet'):
                        new_df=final_df.dropna(subset=[facet_column_col , facet_row_col])

                        #Checking the length of unique values of facet row and col if greater than 3 then we will not the graph
                        temp_row=new_df[facet_row_col].value_counts()
                        row_size=temp_row.shape[0]
                        temp_col=new_df[facet_column_col].value_counts()
                        col_size=temp_col.shape[0]

                        if (row_size>3 and col_size>3):
                            st.warning('Facet Row and Columns have large number of unique values , try using different columns')
                        elif row_size>3:
                            st.warning('Facet row is not present because of high number of unique values in it , choose another row value')
                            fig=px.scatter(new_df, x=x_axis, y=y_axis, facet_col=facet_column_col,color=color_col)
                            st.plotly_chart(fig)
                        elif col_size>3:
                            st.warning('Facet column is not present because of high number of unique values in it , choose another column value')
                            fig=px.scatter(new_df, x=x_axis, y=y_axis, facet_row=facet_row_col,color=color_col)
                            st.plotly_chart(fig)
                        else:
                            fig=px.scatter(new_df, x=x_axis, y=y_axis, facet_row=facet_row_col , facet_col=facet_column_col,color=color_col)    
                            st.plotly_chart(fig)



            if feature=="Pie Charts":
                st.header("Pie Charts")
                try:
                   temp=df.describe()
                   values_col=temp.columns.to_list()
                   x_axis = st.selectbox("Choose a value",values_col)
                   y_axis = st.selectbox("Choose the Category" , col)
                   hover_cols=st.multiselect("hover any data" , col)
                   fig=px.pie(df , values=x_axis , names=y_axis , hover_data=hover_cols)
                   fig.update_traces(textposition='inside' , textinfo='percent+label')
                   if st.button("Plot Pie Chart"):
                      st.plotly_chart(fig)   
                except:
                    st.warning('Try using different columns!')   

            if feature=="Heatmaps, 3D & Ternary Plots":
                st.header("Heatmaps & 3D Plots")
                x_axis = st.selectbox("Choose x-axis", col)
                y_axis = st.selectbox("Choose y-axis" , col)
                z_axis = st.selectbox("Choose z-axis" , col)
                if st.checkbox('Plot Heatmap'):
                    fig = px.density_heatmap(df, x=x_axis, y=y_axis, z=z_axis , marginal_x="histogram", marginal_y="histogram")
                    st.plotly_chart(fig)
                if st.checkbox('Plot 3D graphs'):
                    plot_type=st.selectbox("Choose the plot type" , ['Scatter' , 'Line'])
                    new_col=df.columns.to_list()
                    new_col.insert(0 , '-')
                    color_col=st.selectbox('Choose a color col' , new_col)
                    hover_cols=st.multiselect("Do you want to hover any data" , col)
                    if plot_type=='Scatter':
                        if color_col=='-':
                            fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, hover_data=hover_cols)
                        else:
                            final_df = df.dropna(subset=[color_col])
                            fig = px.scatter_3d(final_df, x=x_axis, y=y_axis, z=z_axis, color=color_col,hover_data=hover_cols)   
                    if plot_type=='Line':
                        if color_col=='-':
                            fig = px.line_3d(df, x=x_axis, y=y_axis, z=z_axis, hover_data=hover_cols)
                        else:
                            final_df = df.dropna(subset=[color_col])
                            fig = px.line_3d(final_df, x=x_axis, y=y_axis, z=z_axis,color=color_col,hover_data=hover_cols)    
                    st.plotly_chart(fig)
                if st.checkbox("Ternary Plot"):
                    temp=df.describe()
                    values_col=temp.columns.to_list()
                    st.write("For Better Visualisation make sure the x , y & z axis are numeric take look at below columns.")
                    st.write(values_col)
                    fig = px.scatter_ternary(df, a=x_axis, b=y_axis, c=z_axis)
                    st.plotly_chart(fig)


        #Inbuilt python Library which generate automated EDA analysis
        elif choice == 'Pandas Profiling Report':
            st.subheader("Automated EDA with Pandas Profiling")
            st.write("Pandas Profiling helps in generating an Automated Exploratory Data Analysis for small Datasets")
            if data is not None:
                if st.button("Generate pandas profiling Report"):
                    try:
                        profile = helper.generate_pandas_profile_report(df)
                        st_profile_report(profile)
                    except:
                        st.warning("Opps!! Something went wrong\nTry again with small data set")    
            
        #Sweetviz report is best to analyse large Datasets
        elif choice=='Sweetviz Report':
            st.subheader("Automated EDA with Sweetviz")
            st.write("Sweetviz helps in generating an Automated Exploratory Data Analysis for small Datasets as well as large Datasets")
            if data is not None:
               if st.button("Generate Sweetviz Report"):
                   try:
                       report = helper.generate_sweetviz_report(df)
                       report.show_html()
                       st_display_sweetviz("SWEETVIZ_REPORT.html")
                   except:
                       st.warning("Opps!! Something went wrong\nTry again with another dataset")    