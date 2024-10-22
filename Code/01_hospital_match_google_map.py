'''
Created by Yanzun Yang
Last edit: 2024/10/21
This file is to find the google maps name for unmatched hospitals after Step 1 in 01_hospital_match.py
'''

import requests
import googlemaps
import pandas as pd

# %%
# Load the list of unmatched hospital names
hospitals_for_gm = merged_df1[merged_df1['_merge'] == 'right_only']['company']  # Adjust this with the path to your file

# Function to query Google Maps API for a hospital name
def get_google_maps_name(hospital_name, api_key):
    url = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={hospital_name}&inputtype=textquery&fields=name&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'candidates' in data and len(data['candidates']) > 0:
            return data['candidates'][0]['name']
    return None

# Google Maps API Key
api_key = 'AIzaSyBnjZCIO5ePkHiurh-3N1aaJ__xI82uHks'

# Create a new dataframe to store the results
results = pd.DataFrame(columns=['Original Name', 'Google Maps Name'])

# %%
# Loop through your list of hospitals and get the Google Maps name for each
for hospital_name in hospitals_for_gm:
    google_maps_name = get_google_maps_name(hospital_name, api_key)
    results = results.append({'Original Name': hospital_name, 'Google Maps Name': google_maps_name}, ignore_index=True)

# Save the results to a CSV file
results.to_csv('Output_data/unmatched_hospital_google_maps_names.csv', index=False)


