import streamlit as st

import pandas as pd
import numpy as np

#import matplotlib.pyplot as plt
#from ipywidgets import widgets

#import mpld3
#from mpld3 import plugins

#import pydeck as pdk
#from pydeck.types import String

from plotly import express as px


st.title('Remote Work Preferences Predictor')


WFH_clean = pd.read_csv("Cleaned_DataFrame.csv", low_memory=False)


                  ####CATEGORICAL FEATURES####
# wfh_days_postCOVID_boss: "Employer's planned number of paid WFH days after COVID"

                    ####CATEGORICAL LABELS####
# numwfh_days_postCOVID_s_u: "Desired share of paid working days WFH after COVID (%)"
# wfh_feel_quant: "How much of a raise/pay cut would you value WFH 2 to 3 days per week? (%)"
# wfh_able_quant: "How efficient are you at working from home? (%)"
# - wfh_coordinate_eff: "Which of the following would make your job more efficient?"
# - wfh_coordinate_pref "Which of the following would you prefer?"
#                        1: "Coworkers coordinate to come in" 
#                        2: "Each coworker decides when to come in" 
#                        3: "No difference"

cat_feat = [ 'education'
            , 'race_ethnicity_s'
            , 'workstatus_current'
            , 'work_industry'
            , 'region'
            , 'gender'
            , 'live_adults'
            , 'live_children'
            , 'wfh_days_postCOVID_boss'
            , 'wfh_days_postCOVID_s'
            , 'wfh_expect']

ranges = { 'education': ["Less than high-school graduation", "High-school graduation", "1 to 3-years of college"
                         , "4 years of college degree", "Masters or Professional Degree", "PhD"]
         , 'race_ethnicity_s': ["Black or African American", "Hispanic (of any race)", "Other", "White (non-Hispanic)"]
         , 'workstatus_current': ["Working on my business premises", "Working from home", "Not working"]
         , 'work_industry': ["Agriculture", "Arts & Entertainment", "Finance & Insurance", "Construction"
                             , "Education", "Health Care & Social Assistance", "Hospitality & Food Services"
                             , "Information", "Manufacturing", "Mining", "Professional & Business Services"
                             , "Real Estate", "Retail Trade", "Transportation and Warehousing", "Utilities"
                             , "Wholesale Trade", "Government", "Other"]
         , 'region': ['AK','AL','AR','AZ','CA','CO','CT','DC', 'DE','FL','GA','HI','IA','ID'
                      ,'IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT'
                      ,'NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI'
                      ,'SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']
         , 'gender': ["Female", "Male", "Other/prefer not to say"]
         , 'live_adults': ["No", "Yes, partner/adult children", "Yes, roommates/other"]
         , 'live_children': ["No", "Yes, youngest in pre-/primary", "Yes, youngest in ES"
                             , "Yes, youngest is in MS", "Yes, youngest is in HS"]
         , 'wfh_days_postCOVID_boss': ["Never", "Rarely", "1 day per week", "2 days per week"
                                       , "3 days per week", "4 days per week", "5 day per week"
                                       , "No clear plans from employer", "No employer"]
         , 'wfh_days_postCOVID_s': ["Never", "Rarely (e.g. monthly)", "1 day per week", "2 days per week"
                                    , "3 days per week", "4 days per week", "5 days per week"]
         , 'wfh_expect': ["Hugely better (>20%)", "Substantially better (10 to 20%)"
                          , "Better (up to 10%)", "About the same ", "Worse (up to 10%)"
                          , "Substantially worse (10 to 20%)", "Hugely worse (>20%)"]
         }



cont_feat = [ 'income'
             , 'age_quant'
             , 'logpop_den_current'
             , 'logpop_den_job_current'
             , 'commutetime_quant'
             , 'wfh_hoursinvest'
             , 'wfhcovid_frac']

all_features = {'education': 'Education'
            , 'race_ethnicity_s': 'Race/Ethnicity'
            , 'workstatus_current': 'Current working status'
            , 'work_industry': 'Work Industry'
            , 'region': 'State of residence'
            , 'gender': 'Gender'
            , 'live_adults': 'Lives with a partner or another adult'
            , 'live_children': 'Lives with children'
            , 'wfh_days_postCOVID_boss': "Employer's planned number of remote work days"
            , 'wfh_days_postCOVID_s': "Desired share of paid working days working from home (%)"
            , 'wfh_expect': 'Efficiency working from home relative to expectations (%)'
            , 'income': 'Income (US$)'
            , 'age_quant': 'Age (years)'
            , 'logpop_den_current': 'Population density of the ZIP code of current residence'
            , 'logpop_den_job_current': 'Population density of the ZIP code of current job business premises'
            , 'commutetime_quant': 'Commute time to work (mins)'
            , 'wfh_hoursinvest': 'Time invested in learning how to work from home effectively (hours)'
            , 'wfhcovid_frac': 'Share of current paid working days working from home (%)'}
key_features = list(all_features.keys())
values_features = list(all_features.values())

labels = { 
    'numwfh_days_postCOVID_s_u': "Desired share of remote work days (% of working days)"
    , 'wfh_feel_quant': "Value of working from home 2 to 3 days per week (% of current salary)"
        }
key_labels = list(labels.keys())
values_labels = list(labels.values())





# User interaction: what is the desired prediction and which variables are available:
label_choice = st.selectbox(
    'What would you like to predict?',
     list(labels.values()))
if label_choice == 'What is your desired share of paid working days working from home after COVID?':
    del all_features['wfh_days_postCOVID_s']

features_choice = st.multiselect(
    'Select all available features',
    list(all_features.values()), ['State of residence']
)

# We collect the data we will use to train our model
label = key_labels[values_labels.index(label_choice)]
features = [key_features[values_features.index(i)] for i in features_choice]
categorical_features = list( set(cat_feat) & set(features) )
continuous_features = list( set(cont_feat) & set(features) )

# Finally, we collect input data
input_data = []
input_dict = {}

st.write('Please enter the data below:')
for f in categorical_features:
    text = st.selectbox(all_features[f], ranges[f])
#    st.write('You selected:', text)
    input_data.append(ranges[f].index(text)+1)
    input_dict[f]=text

for f in continuous_features:
    number = st.number_input(all_features[f], min_value=0)
#    st.write('You selected:', number)
    input_data.append(number)

    
    
    

X = WFH_clean
y = WFH_clean[label]


from models import Model, Averager

model = Model(categorical_features, continuous_features, label)


fitted_predictor = model.fit(X)

prediction = model.predict(np.array(input_data).reshape(1, -1))[0]

if model.test_score() < 0.1:
    st.write('The R2 score of our model with the provided features is ', model.test_score())
    
    if label == 'numwfh_days_postCOVID_s_u':
        st.write('To improve accuracy, try adding:') 
        st.write(all_features['wfh_days_postCOVID_boss'], ', or')
        st.write(all_features['wfhcovid_frac'], ', or')
        st.write(all_features['logpop_den_current'])   
    else:
        st.write('To improve accuracy, try adding:') 
        st.write(all_features['wfh_days_postCOVID_s'], ', or')
        st.write(all_features['logpop_den_current'], ', or')
        st.write(all_features['live_adults'])
        

else:
    if label == 'numwfh_days_postCOVID_s_u':
        st.write('Our model predicts: the desired share is about '
                 , round(prediction), '% of paid working days working from home.')
        st.write('The R2 score of our model with the provided features is ', model.test_score())

    else:  
        st.write('Our model predicts: the value of working from home 2 to 3 times a week is about '
                 , round(prediction), '% of current salary.')
        st.write('The R2 score of our model with the provided features is ', model.test_score())
        
    for f in features:
            Y = X[[f, label]].groupby([f],as_index=False).mean()
            
            if f in cat_feat:
                
                if f == 'region':
                    
                    map_fig = px.choropleth(Y, locations='region',
                        locationmode='USA-states',
                        color = label,
                        color_continuous_scale = px.colors.sequential.YlOrRd,
                        scope="usa",
                        labels={label:'Percentage'})
                    
                    st.plotly_chart(map_fig, use_container_width = True)
                    
                else:
                    #fig = plt.figure(figsize=(5,5))
                    #plt.plot(xentry, yentry, '.', color='k')
                    #plt.xlabel(all_features[f])
                    #plt.ylabel('Average ' + labels[label].lower())
                    
                    #st.pyplot(fig)
                    
                    Y[all_features[f]] = Y[f].apply(lambda x: ranges[f][int(x)-1])
                    Y['selected'] = Y[f].apply(lambda x: True if x==ranges[f].index(input_dict[f])+1 else False)
                    
                    avg_fig = px.bar(Y, x=all_features[f], y=label, color='selected'
                                     ,labels={f: all_features[f], label: 'Average ' + labels[label].lower()})
                    avg_fig.update(layout_showlegend=False)
                    st.plotly_chart(avg_fig, use_container_width = True)
            #else:
             #   fig = plt.figure(figsize=(5,5))
             #   plt.plot(xentry, yentry)
             #   plt.xlabel(all_features[f])
             #   plt.ylabel('Average ' + labels[label].lower())
            
             #   st.pyplot(fig)
       
    
    
    
    
        # st.write('The R2 score of the model with all features included is 0.2. Try adding more data to improve accuracy!')

    
#   st.write('The mean accuracy of our prediction is ', model.test_score(), '. This means the prediction is correct ', round(100*model.test_score()), '% of the time. Try adding more information to improve it!')
    

    
    
    
    
    
    
    
    
#lat_long = {'AK':[61.370716,-152.404419],'AL':[32.806671,-86.791130],'AR':[33.729759,-111.431221]
#            ,'AZ':[34.969704,-92.373123],'CA':[36.116203,-119.681564],'CO':[39.059811,-105.311104]
#            ,'CT':[41.597782,-72.755371],'DC': [38.9072,-77.0369], 'DE':[39.318523,-75.507141]
#            ,'FL':[27.766279,-81.686783]
#            ,'GA':[33.040619,-83.643074],'HI':[21.094318,-157.498337],'IA':[42.011539,-93.210526]
#            ,'ID':[44.240459,-114.478828],'IL':[40.349457,-88.986137],'IN':[39.849426,-86.258278]
 #           ,'KS':[38.526600,-96.726486],'KY':[37.668140,-84.670067],'LA':[31.169546,-91.867805]
  #          ,'MA':[42.230171,-71.530106],'MD':[39.363946,-76.502101],'ME':[44.693947,-69.381927]
   #         ,'MI':[43.326618,-84.536095],'MN':[45.694454,-93.900192],'MO':[38.456085,-92.288368]
    #        ,'MS':[32.741646,-89.678696],'MT':[46.921925,-110.454353],'NC':[35.630066,-79.806419]
     #       ,'ND':[47.528912,-99.784012],'NE':[41.125370,-98.268082],'NH':[43.452492,-71.563896]
      #      ,'NJ':[40.298904,-74.521011],'NM':[34.840515,-106.248482],'NV':[38.313515,-117.055374]
       #     ,'NY':[42.165726,-74.948051],'OH':[40.388783,-82.764915],'OK':[35.565342,-96.928917]
        #    ,'OR':[44.572021,-122.070938],'PA':[40.590752,-77.209755],'RI':[41.680893,-71.511780]
         #   ,'SC':[33.856892,-80.945007],'SD':[44.299782,-99.438828],'TN':[35.747845,-86.692345]
          #  ,'TX':[31.054487,-97.563461],'UT':[40.150032,-111.862434],'VA':[37.769337,-78.169968]
#            ,'VT':[44.045876,-72.710686],'WA':[47.400902,-121.490494],'WI':[44.268543,-89.616508]
 #           ,'WV':[38.491226,-80.954453],'WY':[42.755966,-107.302490]}    
#st.write('The top feature scores are:', model.top_features())


                    #st.pydeck_chart(pdk.Deck(
                    #    map_style=None,
                    #    initial_view_state=pdk.ViewState(
                    #        latitude=39.8,
                     #       longitude=-98.6,
                     #       zoom=3,
                     #       min_zoom=2,
                     #       max_zoom=6,
                    #        pitch=50,
                     #   ),
                    #    layers=[
                    #        pdk.Layer(
                     #          'HexagonLayer',
                    #           data=Y[['lat', 'lon']],
                    #           get_position='[lon, lat]',
                    #           radius=200,
                    #           elevation_scale=4,
                    #           elevation_range=[0, 1000],
                    #           pickable=True,
                    #           extruded=True,
                    #        ),
                    #    ],
                   # ))
    
