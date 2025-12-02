import streamlit as st
import pandas as pd
import numpy as np
import pycountry
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Setup ---
st.set_page_config(page_title="Friend Recommendation System",
                   page_icon="images/network.png",
                   layout="centered")

st.title("Social Media Friend Recommendation System")

# --- Load and preprocess dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("SMUsers.csv")

    # Convert DOB → datetime
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    today = pd.to_datetime('today')
    df['Age'] = round((today - df['DOB']).dt.days / 365).astype('Int64')
    df.drop(columns=['DOB'], inplace=True)

    # Process interests
    df['Interests'] = df['Interests'].str.split(",").apply(
        lambda x: [i.strip().replace(" ", "") for i in x]
    )

    df['Age'] = df['Age'].astype(str)

    # Create tags for similarity computation
    df['Tags'] = (
        df['Gender'] + " " +
        df['Country'] + " " +
        df['Age'] + " " +
        df['Interests'].apply(lambda x: " ".join(x))
    )

    # Final minimal structured dataset
    final_df = df[['UserID', 'Name', 'Tags']].reset_index(drop=True)
    return final_df


final_df = load_data()

# --- User Inputs ---
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

countries = [country.name for country in pycountry.countries]
user_country = st.selectbox("Select your Country:", countries, index=None)

# --- Recommendation Function ---
def recommend(user_name, user_gender, user_DOB, user_interest, user_country):
    today = pd.to_datetime('today')
    age = round((today - pd.to_datetime(user_DOB)).days / 365)

    # Process user interests
    interests_clean = [
        i.strip().replace(" ", "")
        for i in user_interest.split(",")
        if i.strip() != ""
    ]

    # Construct dynamic user tag string
    tag_string = f"{user_gender} {user_country} {age} {' '.join(interests_clean)}"

    # Create temporary dynamic user row
    new_user_row = pd.DataFrame({
        "UserID": ["NEWUSER"],
        "Name": [user_name],
        "Tags": [tag_string]
    })

    # Append to dataset
    temp_df = pd.concat([final_df, new_user_row], ignore_index=True)

    # Vectorization
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(temp_df['Tags']).toarray()
    similarity = cosine_similarity(vectors)

    # Compute similarity of new user
    user_index = len(temp_df) - 1
    distances = similarity[user_index]

    recc_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]  # exclude self

    results = []
    for idx, score in recc_list:
        friend = temp_df.iloc[idx][['UserID', 'Name']]
        results.append(friend.to_dict())

    return results


# --- Run Recommendation ---
if st.button("Recommend Friends"):
    if user_name and user_gender and user_country and user_interest:
        recommendations = recommend(
            user_name, user_gender, user_DOB, user_interest, user_country
        )

        if recommendations:
            st.success("Recommended Friends:")

            # ===== Modern Profile Card UI =====
            for friend in recommendations:
                with st.container():
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
                        ">
                            <div style="
                                width: 60px;
                                height: 60px;
                                border-radius: 50%;
                                background: linear-gradient(135deg, #5f5cff, #a37eff);
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                font-size: 20px;
                                color: white;
                                margin-right: 15px;
                                font-weight: bold;
                            ">
                                {friend['Name'][0].upper()}
                            </div>

                            <div style="flex-grow: 1;">
                                <p style="color: white; font-size: 18px; margin: 0; font-weight: 600;">
                                    {friend['Name']}
                                </p>
                                <p style="color: #b5b5b5; font-size: 14px; margin-top: 4px;">
                                    User ID: {friend['UserID']}
                                </p>
                            </div>

                            <div style="text-align: right;">
                                <a style="
                                    text-decoration: none;
                                    color: #ffffff;
                                    background-color: #4f46e5;
                                    padding: 8px 16px;
                                    border-radius: 8px;
                                    font-size: 14px;
                                " href="#">
                                    View Profile
                                </a>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.warning("No recommendations found.")
    else:
        st.error("Please fill all fields before recommending!")
