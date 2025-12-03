import streamlit as st
import pandas as pd
import numpy as np
import pycountry
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel


# ---------------------------------- PAGE SETUP ----------------------------------
st.set_page_config(
    page_title="Friend Recommendation System",
    page_icon="📡",
    layout="centered"
)

st.title("Social Media Friend Recommendation System")


# ---------------------------------- LOAD DATA ----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("SMUsers.csv")

    required_cols = ["UserID", "Name", "Gender", "Country", "DOB", "Interests"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
    today = pd.to_datetime("today")
    df["Age"] = ((today - df["DOB"]).dt.days / 365).round().astype("Int64")

    df["Gender"] = df["Gender"].fillna("Unknown")
    df["Country"] = df["Country"].fillna("Unknown")
    df["Age"] = df["Age"].fillna(0).astype(str)

    df["Interests"] = (
        df["Interests"]
        .fillna("")
        .astype(str)
        .apply(lambda s: [
            i.strip()
            for i in s.split(",") if i.strip()
        ])
    )

    # Build tags for ML similarity
    df["Tags"] = (
        df["Gender"] + " " +
        df["Country"] + " " +
        df["Age"] + " " +
        df["Interests"].apply(lambda x: " ".join(i.replace(" ", "") for i in x))
    ).str.strip()

    df = df[df["Tags"].str.len() > 0].copy()

    return df.reset_index(drop=True)


try:
    final_df = load_data()
except Exception as e:
    st.error("Failed to load SMUsers.csv.")
    st.exception(e)
    st.stop()


# ---------------------------------- INPUT FIELDS ----------------------------------
user_name = st.text_input("Enter your Name:").strip()

user_gender = st.radio(
    "Select your Gender:",
    ["Male", "Female", "Others", "Prefer not to say"],
    index=None,
    horizontal=True
)

user_DOB = st.date_input(
    "When's your Birthday:",
    min_value=datetime.date(1950, 1, 1),
    max_value=datetime.date.today()
)

user_interest = st.text_input("What are your Interests (comma separated)?")

countries = [c.name for c in pycountry.countries]
user_country = st.selectbox("Select your Country:", countries, index=None)



# ---------------------------------- RECOMMENDER ----------------------------------
def recommend(user_name, user_gender, user_DOB, user_interest, user_country):

    today = pd.to_datetime("today")
    age = int(round((today - pd.to_datetime(user_DOB)).days / 365))

    interests_clean = [
        i.strip()
        for i in user_interest.split(",") if i.strip()
    ]

    user_tag = f"{user_gender} {user_country} {age} {' '.join(i.replace(' ', '') for i in interests_clean)}"

    new_user_df = pd.DataFrame({
        "Tags": [user_tag]
    })

    cv = CountVectorizer(max_features=5000, stop_words="english")
    full_vectors = cv.fit_transform(final_df["Tags"])
    user_vector = cv.transform(new_user_df["Tags"])

    cosine_scores = linear_kernel(user_vector, full_vectors).flatten()

    top_indices = cosine_scores.argsort()[::-1][:5]

    results = []
    for idx in top_indices:
        entry = final_df.iloc[idx]
        results.append(entry)

    return results



# ---------------------------------- RUN BUTTON ----------------------------------
if st.button("Recommend Friends"):
    if not (user_name and user_gender and user_country and user_interest):
        st.error("Please fill all fields before recommending!")
    else:
        try:
            recommendations = recommend(
                user_name, user_gender, user_DOB, user_interest, user_country
            )
        except Exception as e:
            st.error("Error generating recommendations.")
            st.exception(e)
            st.stop()

        st.success("Recommended Friends:")

        # SIMPLE CLEAN TEXT OUTPUT (NO USER ID)
        for friend in recommendations:
            st.write(f"**Name:** {friend['Name']}")
            st.write(f"**Gender:** {friend['Gender']}")
            st.write(f"**Country:** {friend['Country']}")
            st.write(f"**Age:** {friend['Age']}")
            st.write(f"**Interests:** {', '.join(friend['Interests'])}")
            st.write("---")
