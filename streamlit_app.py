# Import necessary libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Show title and description
st.title("⚽ Soccer Analyzer")
st.write("Predict the outcome of soccer matches based on team stats and form.")

# Load the trained model
loaded_model = pickle.load(open('model/trained_model_52.sav', 'rb'))

# Team options for the dropdown
options = ['Burnley', 'Sheffield Utd', 'Everton', 'Brighton', 'Bournemouth', 'Newcastle',
           'Brentford', 'Chelsea', 'Manchester Utd', 'Arsenal', 'Luton', 'Crystal Palace',
           'West Ham', 'Aston Villa', 'Liverpool', 'Tottenham', 'Fulham', 'Wolves',
           'Nottingham', 'Manchester City', 'Leeds', 'Leicester', 'Southampton',
           'Watford', 'Norwich', 'Huddersfield', 'Cardiff', 'West Brom', 'Stoke',
           'Swansea', 'Hull', 'Middlesbrough', 'Sunderland', 'QPR']
round_list = ['1','2','3','4','5'] #theres 38 rounds im too lazy to do em all for now
time_list = ['09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00',]
# Get input from the user
team1 = st.selectbox("Select Home Team", options)
team2 = st.selectbox("Select Away Team", options)
round = st.selectbox("round", round_list)
time = st.selectbox("time, in xx:xx format, disregard the minutes", time_list)

time = int(time.split(':')[0])


#################################################features alert
# Function to find rolling averages and other features for the input teams  change
def find_features(home_team, away_team, round, time):
    df = pd.read_json('data/pl_data_dt.json')
    #need home team latest stats, then away team latest stats
    latest_match_home = df[df['home_team'] == home_team].sort_values(by='date', ascending=False).head(1)
    latest_match_away = df[df['away_team'] == away_team].sort_values(by='date', ascending=False).head(1)
    home_columns = latest_match_home[['home_team_code','home_goals_rolling_avg','home_conceded_goals_rolling_avg','home_shots_rolling_avg',
                                    'home_shots_on_goal_rolling_avg','home_target_ratio_rolling_avg'
                                    ,'home_danger_ratio','home_shot_efficiency']]
    away_columns = latest_match_away[['away_team_code','away_goals_rolling_avg','away_conceded_goals_rolling_avg',
                                    'away_shots_rolling_avg','away_shots_on_goal_rolling_avg','away_conversion_rate_rolling_avg','away_target_ratio_rolling_avg'
                                    ,'away_danger_ratio','away_shot_efficiency']]
    features = pd.concat([home_columns.reset_index(drop=True), away_columns.reset_index(drop=True)], axis=1)
    features['round'] = round
    features['time'] = time
    return features

# Collect features for both teams


# Prepare input data for the model ()
#reorder columns ############################ features alert
feature_columns = ['round', 'time', 'home_team_code', 'away_team_code',
       'home_goals_rolling_avg', 'home_conceded_goals_rolling_avg',
       'home_shots_rolling_avg', 'home_shots_on_goal_rolling_avg',
       'home_target_ratio_rolling_avg', 'away_goals_rolling_avg',
       'away_conceded_goals_rolling_avg', 'away_shots_rolling_avg',
       'away_shots_on_goal_rolling_avg', 'away_conversion_rate_rolling_avg',
       'away_target_ratio_rolling_avg',
       'home_danger_ratio','home_shot_efficiency','away_danger_ratio','away_shot_efficiency']
features = find_features(team1,team2,round,time)
input_data = features[feature_columns]
# Reshape the input data into the required format for the model
#input_data = np.array(input_data).reshape(1, -1)

# Use the model to make predictions
if st.button("Predict"):
    try:
        prediction = loaded_model.predict(input_data)
        #st.write(input_data)
        #st.write(prediction)
        # Display prediction result     1 win 2 away win, 0 draw
        if prediction[0] == 1:
            st.write("Prediction: Home Win")
        elif prediction[0] == 2:
            st.write("Prediction: Away Win")
        else:
            st.write("Prediction: Draw")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")