# main.py

import streamlit as st
import pandas as pd
import numpy as np
import pycountry
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Setup page ---
st.set_page_config(page_title="Friend Recommendation System", layout="centered")
st.title("Social Media Friend Recommendation System")

# --- Caching the heavy operations ---
@st.cache_data
def load_data():
    df = pd.read_csv("SMUsers.csv")

    # Preprocessing
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    today = pd.to_datetime('today')
    df['Age'] = round((today - df['DOB']).dt.days / 365).astype('Int64')

    df = df[df['Country'] == 'India'].copy()
    df.drop(columns=['DOB'], inplace=True)

    df['Interests'] = df['Interests'].str.split(", ").apply(lambda x: [i.strip("'").replace(" ", "") for i in x])
    df['Age'] = df['Age'].astype(str)

    df['Tags'] = df['Gender'] + " " + df['City'] + " " + df['Age'] + " " + df['Interests'].apply(lambda x: ' '.join(x))

    final_df = df[['UserID', 'Name', 'Tags']].reset_index(drop=True)

    # Vectorize once
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(final_df['Tags']).toarray()
    similarity = cosine_similarity(vectors)

    return final_df, similarity

final_df, similarity = load_data()

# --- User Input ---
user_name = st.text_input("Enter your Name:")

user_gender = st.radio(
    "Select your Gender:",
    ["Male", "Female", "Others", "Prefer not to say"],
    index=None,
    horizontal=True
)

# Wider range for birthday
user_DOB = st.date_input(
    "When's your Birthday:",
    min_value=datetime.date(1950, 1, 1),
    max_value=datetime.date.today()
)

user_interest = st.text_input("What are your Interests (comma separated)?")

countries = [country.name for country in pycountry.countries]
user_country = st.selectbox("Select your Country:", countries)

# --- Recommend Function ---
def recommend(user_name, user_gender, user_DOB, user_interest, user_country):
    try:
        person_index = final_df[final_df['Name'] == user_name].index[0]
        distances = similarity[person_index]
        recc_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        results = []
        for i in recc_list:
            user_info = final_df.iloc[i[0]][['UserID', 'Name']]
            results.append(user_info.to_dict())

        return results

    except IndexError:
        return None

# --- Button ---
if st.button("Recommend Friends"):
    if user_name and user_interest and user_country:
        recommendations = recommend(user_name, user_gender, user_DOB, user_interest, user_country)
        if recommendations:
            st.success("Recommended Friends:")
            for friend in recommendations:
                st.write(f"{friend['Name']} (UserID: {friend['UserID']})")
        else:
            st.warning("No recommendations found. Please make sure the name exists in database.")
    else:
        st.error("Please fill all fields before recommending!")
