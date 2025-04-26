import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import pycountry


@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("SMUsers.csv")
    df['DOB'] = pd.to_datetime(df['DOB'])
    today = pd.to_datetime('today')
    df['Age'] = round((today - df['DOB']).dt.days / 365).astype(int)
    df = df[df['Country'] == 'India']
    df.drop(columns=['DOB'], inplace=True)
    df['Interests'] = df['Interests'].str.split(
        ", ").apply(lambda x: [i.strip("'") for i in x])
    df['Interests'] = df['Interests'].apply(
        lambda x: [i.replace(" ", "") for i in x])
    df['Age'] = df['Age'].astype(str)
    df['Tags'] = df['Gender'] + " " + df['City'] + " " + df['Age'] + \
        " " + df['Interests'].apply(lambda x: ' '.join(x)) + " "
    final_df = df[['UserID', 'Name', 'Tags']].reset_index(drop=True)

    # Vectorize tags
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(final_df['Tags']).toarray()

    # Calculate similarity matrix
    similarity = cosine_similarity(vectors)

    return final_df, cv, vectors, similarity


final_df, cv, vectors, similarity = load_and_prepare_data()


def recommend(user_name, user_gender, user_DOB, user_interest, user_country):
    today = pd.to_datetime('today')
    age = round((today - pd.to_datetime(user_DOB)).days / 365)

    # Clean interests
    user_interest_cleaned = [i.strip().replace(" ", "")
                             for i in user_interest.split(",")]

    # Create Tags
    user_tags = user_gender + " " + user_country + " " + \
        str(age) + " " + ' '.join(user_interest_cleaned)

    # Transform the new user
    user_vector = cv.transform([user_tags]).toarray()

    # Calculate similarity with existing users
    user_similarity = cosine_similarity(user_vector, vectors).flatten()

    # Get top 5 recommendations
    recc_list = sorted(list(enumerate(user_similarity)),
                       reverse=True, key=lambda x: x[1])[0:5]

    recommendations = []
    for i in recc_list:
        user_id = int(final_df.iloc[i[0]]['UserID'])
        name = final_df.iloc[i[0]]['Name']
        recommendations.append((user_id, name))

    return recommendations


st.title("Social Media Friend Recommendation System")

user_name = st.text_input("Enter your Name:")
user_gender = st.radio(
    "Select your Gender:",
    ["Male", "Female", "others", "prefer not to say"],
    index=None,
    horizontal=True,
)

user_DOB = st.date_input(
    "When's your birthday",
    value=datetime.date(2000, 1, 1),
    min_value=datetime.date(1900, 1, 1),
    max_value=datetime.date.today()
)

user_interest = st.text_input("What are your interests? (comma-separated):")

countries = [country.name for country in pycountry.countries]
user_country = st.selectbox("Select your Country", countries)

if st.button("Recommend Friends"):
    if user_name and user_gender and user_interest:
        recommendations = recommend(
            user_name, user_gender, user_DOB, user_interest, user_country)
        if recommendations:
            st.success("Recommended Friends:")
            for friend in recommendations:
                st.write(f"UserID: {friend[0]} | Name: {friend[1]}")
        else:
            st.warning("No recommendations found for this user.")
    else:
        st.error("Please fill all the fields!")
