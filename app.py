from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and LabelEncoder
with open('best_xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# with open('label_encoder.pkl', 'rb') as label_file:
#     label_encoder = pickle.load(label_file)
# Load encoders
with open('batting_team_encoder.pkl', 'rb') as file:
    batting_team_encoder = pickle.load(file)

with open('balling_team_encoder.pkl', 'rb') as file:
    balling_team_encoder = pickle.load(file)

with open('venue_encoder.pkl', 'rb') as file:
    venue_encoder = pickle.load(file)

with open('pitch_type_encoder.pkl', 'rb') as file:
    pitch_type_encoder = pickle.load(file)

with open('season_econder.pkl','rb') as file:
    season_encoder=pickle.load(file)
# Home page route
@app.route('/')
def home():
    # Sample data for dropdowns
    batting_teams = ['CSK', 'MI', 'PBKS', 'DC', 'KKR', 'RCB', 'SRH', 'RR', 'RPS', 'GT', 'LSG']
    bowling_teams = ['MI', 'CSK', 'DC', 'PBKS', 'RCB', 'KKR', 'RR', 'SRH', 'RPS', 'GT', 'LSG']
    venues = ['Wankhede Stadium', 'Punjab Cricket Association IS Bindra Stadium', 'Eden Gardens', 'Rajiv Gandhi International Stadium',
       'MA Chidambaram Stadium', 'Sawai Mansingh Stadium', 'M.Chinnaswamy Stadium', 'Maharashtra Cricket Association Stadium',
       'Arun Jaitley Stadium', 'Holkar Cricket Stadium', 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
       'Saurashtra Cricket Association Stadium', 'Green Park', 'Sheikh Zayed Stadium', 'Dubai International Cricket Stadium',
       'Sharjah Cricket Stadium', 'Narendra Modi Stadium', 'Brabourne Stadium', 'Dr DY Patil Sports Academy',
       'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium', 'Barsapara Cricket Stadium', 'Himachal Pradesh Cricket Association Stadium',
       'Maharaja Yadavindra Singh International Cricket Stadium', 'Wanderers Stadium', 'SuperSport Park, Centurion',
       'Kingsmead Cricket Ground', 'Mangaung Oval', "St George's Park", 'De Beers Diamond Oval', 'Buffalo Park', 'Newlands Cricket Ground',
       'Vidarbha Cricket Association (VCA) Stadium', 'Barabati Stadium', 'Jawaharlal Nehru International Stadium',
       'JSCA International Stadium Complex', 'Shaheed Veer Narayan Singh International Cricket Stadium']
    pitch_types = ['Batting-Friendly', 'Balanced', 'Bowling-Friendly']
    return render_template('index.html',
                           batting_teams=batting_teams,
                           bowling_teams=bowling_teams,
                           venues=venues,
                           pitch_types=pitch_types)
# Function to check if the venue is a home ground
def is_home_ground(row, team_column):
    # Define home venues for each team
    home_venues = {
        'MI': ['Wankhede Stadium'],
        'CSK': ['MA Chidambaram Stadium'],
        'DC': ['Arun Jaitley Stadium'],
        'PBKS': ['Punjab Cricket Association IS Bindra Stadium'],
        'RCB': ['M.Chinnaswamy Stadium'],
        'KKR': ['Eden Gardens'],
        'RR': ['Sawai Mansingh Stadium'],
        'SRH': ['Rajiv Gandhi International Stadium'],
        'GT': ['Narendra Modi Stadium'],
        'LSG': ['Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium']
    }
    team = row[team_column]
    venue = row['Venue']
    return 1 if team in home_venues and venue in home_venues[team] else 0

# Function to categorize months into Winter, Summer, and Monsoon
def categorize_season(month):
    if month in [12, 1, 2]:  # Winter months
        return "Winter"
    elif month in [3, 4, 5]:  # Summer months
        return "Summer"
    else:  # Monsoon months
        return "Monsoon"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        batting_team = request.form['Batting Team']
        balling_team = request.form['Balling Team']
        venue = request.form['Venue']
        match_date = request.form['Match Date']  # This may need to be processed if it's part of your features
        over_number = int(request.form['Over number'])
        score_after_over = float(request.form['score after over'])
        wickets_after_over = int(request.form['wickets after over'])
        pitch_type = request.form['Pitch Type']  # Added Pitch Type field
        print("printing pitch type ", pitch_type)
        run_rate=float(score_after_over/over_number)

        # Convert match_date to datetime64[ns]
        match_date = pd.to_datetime(match_date, format='%Y-%m-%d')

        # Extract features from match_date
        year = match_date.year
        month = match_date.month
        day = match_date.day
        day_of_week = match_date.dayofweek  # Monday = 0, Sunday = 6

        # Print the extracted values for debugging
        print(f"Year: {year}")
        print(f"Month: {month}")
        print(f"Day: {day}")
        print(f"Day of Week: {day_of_week}")
        Day_Sin = np.sin(2 * np.pi * day / 31)
        Day_Cos = np.cos(2 * np.pi * day/ 31)

        # Apply the function to create a new 'Season' column
        Season = categorize_season(month)
        # Prepare the input data without the Match Date
        input_data = pd.DataFrame({
            'Batting Team': [batting_team],
            'Balling Team': [balling_team],
            'Venue': [venue],
            'Over number': [over_number],
            'score after over': [score_after_over],
            'wickets after over': [wickets_after_over],
            'Year': [year],
            'Month': [month],
            'Day': [day],
            'Day_Sin':[Day_Sin],
            'Day_Cos':[Day_Cos],
            # 'Day of Week': [day_of_week],
            'Run Rate':[run_rate],
            'Batting Home Ground':[0],
            'Balling Home Ground':[0],
            'Pitch Type': [pitch_type],
            'Season':[Season]
        })
        # Add columns for batting and bowling home ground (1 or 0)
        input_data['Batting Home Ground'] = input_data.apply(lambda row: is_home_ground(row, 'Batting Team'), axis=1)
        input_data['Balling Home Ground'] = input_data.apply(lambda row: is_home_ground(row, 'Balling Team'), axis=1)


        # Label encode categorical features with error handling
        print("going into encoders")
        # Label encode categorical features
        input_data['Batting Team'] = batting_team_encoder.transform(input_data['Batting Team'])
        input_data['Balling Team'] = balling_team_encoder.transform(input_data['Balling Team'])
        input_data['Venue'] = venue_encoder.transform(input_data['Venue'])
        input_data['Pitch Type'] = pitch_type_encoder.transform(input_data['Pitch Type'])
        input_data['Season']=season_encoder.transform(input_data['Season'])
        # Make prediction using the loaded model
        predicted_inning_score = model.predict(input_data)

        # Render the result on the home page
        return render_template('index.html',
                               prediction_text='Predicted Inning Score: {:.2f}'.format(predicted_inning_score[0]))

    except ValueError as e:
        # Handle label encoding or conversion errors
        return render_template('index.html',
                               error_message=f"Error: {str(e)}. Please ensure the input values are correct.")

    except Exception as e:
        # Handle any other unexpected errors
        return render_template('index.html',
                               error_message=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
