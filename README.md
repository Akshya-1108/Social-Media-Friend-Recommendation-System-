# Social Media Friend Recommendation System

A lightweight Streamlit web application that recommends potential friends based on a user's profile inputs like **name**, **gender**, **date of birth**, **country**, and **interests**.

This app uses basic **natural language processing (NLP)** and **cosine similarity** to find users with similar tags and recommend them as friends.

---

## Features

- 🖋️ Streamlit-based simple and clean UI.
- 🔱 Dynamic input fields for:
  - Name
  - Gender (Male, Female, Others)
  - Date of Birth (with wide year range selection)
  - Country (select from all countries)
  - Interests (comma-separated)
- ✨ Fast friend recommendations based on user profile.
- 🌐 Optimized with caching to avoid redundant heavy calculations.
- 📊 Uses `CountVectorizer` + `Cosine Similarity` for matching user interests.

---

## Workflow

1. **Load User Data**: Load existing user data from `SMUsers.csv`.
2. **Preprocess**: Clean and vectorize data using NLP techniques.
3. **Cache Processing**: Heavy operations like vectorization and similarity matrix are cached.
4. **User Input**: User fills in their profile details.
5. **Generate Tags**: Create a profile tag string from user input.
6. **Find Recommendations**: Calculate cosine similarity and recommend top 5 closest profiles.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/social-media-friend-recommendation.git

# Navigate to the project folder
cd social-media-friend-recommendation

# Install required packages
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- pycountry

---

## Folder Structure

```bash
/
|-- app.py            # Main Streamlit app file
|-- SMUsers.csv       # User dataset
|-- requirements.txt  # Required Python packages
|-- README.md         # Project overview
```

---

## Future Improvements

- 🌈 Profile cards UI (user image, short bio)
- 🕷️ Interest keyword analysis and clustering
- 🛍️ Friend request simulation system
- 💸 Deploy on cloud (Streamlit Share, HuggingFace Spaces, or AWS)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

**Made with ❤️ for social connection and machine learning exploration!**

