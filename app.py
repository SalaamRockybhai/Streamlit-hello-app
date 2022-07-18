import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np




# Load all the pickle files
# onehotencoder = pickle.load(open('models/onehotencoder.pkl','rb'))
# lablencoder1  = pickle.load(open('models/lablencoder1.pkl','rb'))
# lablencoder2  = pickle.load(open('models/lablencoder2.pkl','rb'))
# scaler        = pickle.load(open('models/scaler.pkl','rb'))
# regressor     = joblib.load(os.path.expanduser('models/regressor.joblib'))
# with open('models/regressor.joblib', "rb") as f:  
#      regressor = joblib.load(f)



# Load the playstore data from google drive 
df12 = pd.read_csv('Car_price_prediction_clean.csv')


st.write('# Car Price Predictor')


gif = 'https://cutewallpaper.org/24/driving-a-car-animated-gif/man-driving-car-stock-animation-car-animation-2d-character-animation-cute-cartoon-wallpapers.gif'
st.image(
          gif, # I prefer to load the GIFs using GIPHY
          width=200, # The actual size of most gifs on GIPHY are really small, and using the column-width parameter would make it weirdly big. So I would suggest adjusting the width manually!
     )

with st.sidebar:

     st.write('#### Please enter all the information of your car')
     # Levy input
     levy = st.slider("How much is the levy (export tax)?", 0, 2300, 0)

     # manufacturer
     manufacturer = st.selectbox(
          label   = "Who is your car's manufacturer.",
          options = df12['manufacturer'].value_counts().index
          )

     #  st.write('You selected:', manufacturer)


     # model
     model = st.selectbox(
          label   = "What is your car's Model.",
          options = df12['model'].value_counts().index
          )

     #  st.write('You selected:', Model)


     # Prod. year
     pyear = st.selectbox(
          label   = "What is your car's production year.",
          options = sorted(df12['prd_yr'].unique().tolist(), reverse=True) 
          )

     # st.write('You selected:', pyear)

     # Category
     category = st.selectbox(
          label   = "What is the category of your car?",
          options = df12['category'].value_counts().index
          )

     #  st.write('You selected:', Category)

     # Leather interior

     leatherintr = st.radio(
          "Do your car has leather interior?",
          ('Yes', 'No'))

     if leatherintr ==  'Yes':
          leather_intr = 1
     else:    
          leather_intr = 0

     #  st.write('You selected:', leather_intr)

     # fuel_typ
     fuel_typ = st.selectbox(
          label   = "What is the fuel type of your car?",
          options = df12['fuel_typ'].value_counts().index
          )

     #  st.write('You selected:', fuel_typ)


     # engine_vol input
     engine_vol = st.slider(label    = "What is the engine volume of your car?", 
                         min_value=round(df12['engine_vol'].min(),1), 
                         max_value=round(df12['engine_vol'].max(),1), 
                         value=2.5)

     # Turbo engine
     engineturbo = st.radio(
          "Do your car has turbo engine?",
          ('Yes', 'No'),1)

     if engineturbo ==  'Yes':
          engine_turbo = 1
     else:    
          engine_turbo = 0


     # mileage input
     mileage = st.slider(label    = "What is the mileage (in km) of your car?", 
                         min_value   = round(df12['mileage(km)'].min(),1), 
                         max_value   = round(df12['mileage(km)'].max(),1), 
                         value       = float(round(df12['mileage(km)'].mean(),1))
                         )


     # gear_box
     gear_box = st.selectbox(
          label   = "What is the gear box type of your car?",
          options = df12['gear_box'].value_counts().index
          )

     # gear_box
     drive_wheels = st.selectbox(
          label   = "What is the drive wheel type of your car?",
          options = df12['drive_wheels'].value_counts().index
          )

     # doors
     doors = st.selectbox(
          label   = "How many doors are there in your car?",
          options = df12['doors'].value_counts().index
          )

     # wheel
     wheel = st.selectbox(
          label   = "What is the wheels type in your car?",
          options = df12['wheel'].value_counts().index
          )

     # wheel
     color = st.selectbox(
          label   = "What is the color of your car?",
          options = df12['color'].value_counts().index
          )






     # airbags input
     airbags = st.slider(label    = "How many air bags are there in your car?", 
                         min_value   = int(round(df12['airbags'].min(),0)), 
                         max_value   = int(round(df12['airbags'].max(),0)), 
                         value       = int(round(df12['airbags'].mean(),0))
                         )

          
          
          
          

col = ['levy', 'manufacturer', 'model', 'prd_yr', 'category', 'leather_intr',
       'fuel_typ', 'engine_vol', 'mileage(km)', 'gear_box', 'drive_wheels',
       'doors', 'wheel', 'color', 'airbags', 'engine_turbo']

values = [[levy, manufacturer, model, pyear, category, leather_intr,
       fuel_typ, engine_vol, mileage, gear_box, drive_wheels,
       doors, wheel, color, airbags, engine_turbo]]

X_test = pd.DataFrame(values, columns = col)

             
             
                    

try:
    regressor     = joblib.load('Newreg.pkl')
except:
    st.write('Newreg files are unable to load')

try:
    # Code to test / execute
    onehotencoder = joblib.load('onehotencoder.pkl')
    lablencoder1  = joblib.load('lablencoder1.pkl')
    lablencoder2  = joblib.load('lablencoder2.pkl')
    scaler        = joblib.load('scaler.pkl')
except:
    st.write('pickle files are unable to load')




try:
    X_test_OHE             = onehotencoder.transform(X_test[['category','fuel_typ','gear_box','drive_wheels','doors','wheel','color']])
    X_test['manufacturer'] = lablencoder1.transform(X_test['manufacturer'])
    X_test['model']        = lablencoder2.transform(X_test['model'])

    X_test_rem             = X_test.drop(['category','fuel_typ','gear_box','drive_wheels','doors','wheel','color'], axis =1 )
    X_test_transformed     = np.concatenate((X_test_rem , X_test_OHE) ,axis=1)
    X_test_scale  = scaler.transform(X_test_transformed)
    prize = round(regressor.predict(X_test_scale)[0])

    if st.button('Predict Price'):
        st.write('### The estimated price of your car is just $' + str(prize) + "...!")

except:
    st.write('Unable to predict.')
