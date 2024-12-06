# Import necessary libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np
#streamlit run streamlit_app.py to run

# Show title and description
st.title("⚽ Soccer Analyzer")
st.write("Predict the outcome of soccer matches based on team stats and form.")

# Load the trained model
loaded_model = pickle.load(open('model/trained_model_52.sav', 'rb'))
loaded_goal_model = pickle.load(open('model/trained_model_goal.sav','rb'))
quantile_model = pickle.load(open('model/quantile_model.sav', 'rb'))
quantile_model_away = pickle.load(open('model/quantile_model_away.sav', 'rb'))

# Team options for the dropdown
options = ['Burnley', 'Sheffield Utd', 'Everton', 'Brighton', 'Bournemouth', 'Newcastle',
           'Brentford', 'Chelsea', 'Manchester Utd', 'Arsenal', 'Luton', 'Crystal Palace',
           'West Ham', 'Aston Villa', 'Liverpool', 'Tottenham', 'Fulham', 'Wolves',
           'Nottingham', 'Manchester City', 'Leeds', 'Leicester', 'Southampton',
           'Watford', 'Norwich', 'Huddersfield', 'Cardiff', 'West Brom', 'Stoke',
           'Swansea', 'Hull', 'Middlesbrough', 'Sunderland', 'QPR']
round_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30'] #theres 38 rounds im too lazy to do em all for now
time_list = ['12:30','14:00','15:00','16:30','17:30',]
# Select Away Team first
team1 = st.selectbox("Select Home Team", options)
team2_options = [team for team in options if team != team1]  # Exclude selected team1
team2 = st.selectbox("Select Away Team", team2_options)

round = int(st.selectbox("round", round_list))
time = st.selectbox("time", time_list)

time = int(time.split(':')[0])


#################################################features alert
#incomplete read below pls tyvm
# note to self try and find the row where the team stats for rolling averages are on for the inputted round as well as time
# Function to find rolling averages and other features for the input teams  change
def find_features(home_team, away_team, round, time):
    df = pd.read_json('data/pl_data_dt.json')
    #need home team latest stats, then away team latest stats
    latest_match_home = df[df['home_team'] == home_team].sort_values(by='date', ascending=False).head(1)
    latest_match_away = df[df['away_team'] == away_team].sort_values(by='date', ascending=False).head(1)
    home_columns = latest_match_home[['home_team_code','home_goals_rolling_avg','home_conceded_goals_rolling_avg','home_shots_rolling_avg',
                                    'home_shots_on_goal_rolling_avg','home_target_ratio_rolling_avg'
                                    ,'home_danger_ratio','home_shot_efficiency','home_conversion_rate_rolling_avg'
                                    ,'home_target_to_goal_ratio_rolling_avg'
                                    ,'home_shot_creation_ratio_rolling_avg'
                                    ,'home_shots_off_goal_rolling_avg']]
    away_columns = latest_match_away[['away_team_code','away_goals_rolling_avg','away_conceded_goals_rolling_avg',
                                    'away_shots_rolling_avg',
                                    'away_shots_on_goal_rolling_avg',
                                    'away_conversion_rate_rolling_avg',
                                    'away_target_ratio_rolling_avg'
                                    ,'away_danger_ratio',
                                    'away_shot_efficiency'
                                    ,'away_target_to_goal_ratio_rolling_avg'
                                    ,'away_shot_creation_ratio_rolling_avg'
                                    ,'away_shots_off_goal_rolling_avg']]
    features = pd.concat([home_columns.reset_index(drop=True), away_columns.reset_index(drop=True)], axis=1)
    features['round'] = round
    features['time'] = time
    return features

# Collect features for both teams


# Prepare input data for the model () redo model with correct corresponding features
#reorder columns ############################ features alert
feature_columns = ['round', 'time', 'home_team_code', 'away_team_code',
        'home_goals_rolling_avg',
        'home_conceded_goals_rolling_avg',
        'home_shots_rolling_avg',
        'home_shots_on_goal_rolling_avg',
        'home_target_ratio_rolling_avg', 
        'home_conversion_rate_rolling_avg',
       'away_goals_rolling_avg',
       'away_conceded_goals_rolling_avg', 
       'away_shots_rolling_avg',
       'away_shots_on_goal_rolling_avg', 
       'away_conversion_rate_rolling_avg',
       'away_target_ratio_rolling_avg',
       'home_danger_ratio','home_shot_efficiency','away_danger_ratio','away_shot_efficiency']

goal_features_home = ['home_shots_rolling_avg',
                      'home_shots_on_goal_rolling_avg', 
                      'home_target_to_goal_ratio_rolling_avg', 
                      'home_conversion_rate_rolling_avg',
                      'home_shot_creation_ratio_rolling_avg', 
                      'home_shots_off_goal_rolling_avg']
goal_features_away = ['away_shots_rolling_avg',
                      'away_shots_on_goal_rolling_avg', 
                      'away_target_to_goal_ratio_rolling_avg', 
                      'away_conversion_rate_rolling_avg',
                      'away_shot_creation_ratio_rolling_avg', 
                      'away_shots_off_goal_rolling_avg']
features = find_features(team1,team2,round,time)
input_data = features[feature_columns]
input_data_goals_home = features[goal_features_home]
input_data_goals_away = features[goal_features_away]

# Reshape the input data into the required format for the model
input_data_goals_home = np.array(input_data_goals_home).reshape(1, -1)
input_data_goals_away = np.array(input_data_goals_away).reshape(1, -1)
#displaying to doublecheck
#st.write(input_data)
#st.write(input_data_goals_home)
#st.write(input_data_goals_away)
def predict_goal_range(model, input_data):
    try:
        y_pred = model.predict(input_data, quantiles=[0.16, 0.84])
        predicted_left_goals = int(np.floor(y_pred[0, 0]))
        predicted_high_goals = int(np.ceil(y_pred[0, 1]))
        return predicted_left_goals, predicted_high_goals
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        return None, None


# Use the model to make predictions
if st.button("Predict Match Win"):
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
            st.write("Prediction: Inconclusive, not enough matches at this time to determine")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if st.button("Predict Goal Range for Home Team"):
    left_goals, high_goals = predict_goal_range(quantile_model, input_data_goals_home)
    if left_goals is not None and high_goals is not None:
        st.write(f"The home team will score {left_goals} (predicted left goals) - {high_goals} (predicted high goals) goals.")

# New button for predicting goal range for Away Team
if st.button("Predict Goal Range for Away Team"):
    left_goals, high_goals = predict_goal_range(quantile_model_away, input_data_goals_away)
    if left_goals is not None and high_goals is not None:
        st.write(f"The away team will score {left_goals} (predicted left goals) - {high_goals} (predicted high goals) goals.")

#away team not done yet