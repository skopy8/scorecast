import streamlit as st
import pickle
import pandas as pd

# Show title and description. any message in write
st.title("💬 Soccer Analyzer")
st.write(
    
)
#load model and features
loaded_model = pickle.load(open('model/trained_model.sav', 'rb'))
options = ['Burnley', 'Sheffield Utd', 'Everton', 'Brighton', 'Bournemouth', 'Newcastle',
 'Brentford', 'Chelsea' ,'Manchester Utd', 'Arsenal' ,'Luton' ,'Crystal Palace',
 'West Ham', 'Aston Villa', 'Liverpool' ,'Tottenham', 'Fulham', 'Wolves',
 'Nottingham', 'Manchester City', 'Leeds', 'Leicester', 'Southampton',
 'Watford', 'Norwich', 'Huddersfield' ,'Cardiff', 'West Brom', 'Stoke',
 'Swansea' ,'Hull' ,'Middlesbrough', 'Sunderland', 'QPR']
#get input from user
st.selectbox("team1", options)
st.selectbox("team2", options)



#using model
if st.button("Predict"):
    input_data = [[]]  # Add other features
    prediction = loaded_model.predict(input_data)
    st.write("Prediction:", prediction)




#function for finding relevant features from our dataset by given team (latest rolling averages and etc)
def findFeatures(team):
    df = pd.read_json('data/pl_data_dt.json')
    df['date'] = pd.to_datetime(df['date'])
    team_data = df[(df['home_team'] == team)]
    team_data = team_data.sort_values(by='date', ascending=False)







#stopping here for now, in the middle of working on findfeatures function, cant progress (cant be arsed to progress)
# findfeatures needs to find the latest rolling averages given the team input for that particular team
#then find it for team2, append the two feature lists so its one dictionary so we can input it into input_data
#then just use predict model.





# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management