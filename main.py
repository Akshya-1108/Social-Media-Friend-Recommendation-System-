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

    # Validate dataset
    required_cols = ["UserID", "Name", "Gender", "Country", "DOB", "Interests"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # DOB → Age
    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
    today = pd.to_datetime("today")
    df["Age"] = ((today - df["DOB"]).dt.days / 365).round().astype("Int64")

    # Handle missing
    df["Gender"] = df["Gender"].fillna("Unknown")
    df["Country"] = df["Country"].fillna("Unknown")
    df["Age"] = df["Age"].fillna(0).astype(str)

    # Clean Interests
    df["Interests"] = (
        df["Interests"]
        .fillna("")
        .astype(str)
        .apply(lambda s: [
            i.strip().replace(" ", "")
            for i in s.split(",") if i.strip() != ""
        ])
    )

    # Build Tags
    df["Tags"] = (
        df["Gender"] + " " +
        df["Country"] + " " +
        df["Age"] + " " +
        df["Interests"].apply(lambda x: " ".join(x))
    ).str.strip()

    # Remove empty tag rows
    df = df[df["Tags"].str.len() > 0].copy()

    final_df = df[["UserID", "Name", "Tags"]].reset_index(drop=True)
    return final_df


# Load dataset
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



# ---------------------------------- SCALABLE RECOMMENDER ----------------------------------
def recommend(user_name, user_gender, user_DOB, user_interest, user_country):

    # Compute Age
    today = pd.to_datetime("today")
    age = int(round((today - pd.to_datetime(user_DOB)).days / 365))

    # Clean interests
    interests_clean = [
        i.strip().replace(" ", "")
        for i in user_interest.split(",")
        if i.strip() != ""
    ]

    # User vector text
    user_tag = f"{user_gender} {user_country} {age} {' '.join(interests_clean)}".strip()

    new_user_df = pd.DataFrame({
        "UserID": ["NEWUSER"],
        "Name": [user_name],
        "Tags": [user_tag]
    })

    # Fit vectorizer on full dataset
    cv = CountVectorizer(max_features=5000, stop_words="english")
    full_vectors = cv.fit_transform(final_df["Tags"])  # sparse (100k x 5k)

    # Transform only new user
    user_vector = cv.transform(new_user_df["Tags"])  # sparse (1 x 5k)

    # Compute similarity: 1 × 100k (very small memory)
    cosine_scores = linear_kernel(user_vector, full_vectors).flatten()

    # Top 5 matches
    top_indices = cosine_scores.argsort()[::-1][:5]

    results = []
    for idx in top_indices:
        person = final_df.iloc[idx][["UserID", "Name"]]
        results.append(person.to_dict())

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

        # Attractive UI Cards
        for friend in recommendations:
                st.markdown(
    f"""
    <div style="
        background-color: #1e1e1e;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 12px;
        border: 1px solid #333;
        display: flex;
        align-items: center;
        gap: 16px;
    ">
        
        <!-- Avatar -->
        <div style="
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #5f5cff, #a37eff);
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 24px;
            font-weight: 700;
            color: white;
            text-transform: uppercase;
        ">
            {friend['Name'][0]}
        </div>

        <!-- Name + User ID -->
        <div style="flex-grow: 1;">
            <p style="color: white; font-size: 18px; margin: 0; font-weight: 600;">
                {friend['Name']}
            </p>
            <p style="color: #b5b5b5; font-size: 14px; margin: 4px 0 0;">
                User ID: {friend['UserID']}
            </p>
        </div>

        <!-- Button -->
        <div>
            <a href="#" style="
                text-decoration: none;
                color: #ffffff;
                background-color: #4f46e5;
                padding: 8px 16px;
                border-radius: 8px;
                font-size: 14px;
            ">
                View Profile
            </a>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)
