import streamlit as st
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import requests
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import re
from langdetect import detect
from googleapiclient.discovery import build

# Google OAuth 2.0 credentials
CLIENT_ID = "829529311962-0c97m8n8bmm05668cambtqqsn6nlbm4c.apps.googleusercontent.com"
CLIENT_SECRET = "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"
REDIRECT_URI = "http://localhost:8501"

SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtubepartner"
]

TOKEN_FILE = "token.json"
TRAIN_DATA_FILE = "TrainBot 2.csv"


def save_credentials(credentials):
    """Save user credentials to a file."""
    with open(TOKEN_FILE, "w") as token:
        json.dump({
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes
        }, token)


def load_credentials():
    """Load user credentials from a file."""
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as token:
            creds_data = json.load(token)
            return Credentials.from_authorized_user_info(creds_data, SCOPES)
    return None


def authenticate_user():
    """Generate a Google authentication URL."""
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI]
            }
        },
        scopes=SCOPES,
    )

    flow.redirect_uri = REDIRECT_URI

    auth_url, _ = flow.authorization_url(prompt="consent")
    st.write("Please log in using Google by clicking the link below:")
    st.markdown(f"[Login to Google]({auth_url})")

    # Capture the authorization code
    auth_code = st.text_input("Paste the authorization code here:")

    if auth_code:
        try:
            flow.fetch_token(code=auth_code)
            credentials = flow.credentials
            save_credentials(credentials)
            st.success("Login successful! Reloading the app...")

            # Workaround to refresh the app state
            st.experimental_set_query_params(reload="true")
        except Exception as e:
            st.error(f"Error during login: {e}")


def fetch_youtube_data(credentials, video_id):
    """Fetch comments from a YouTube video."""
    headers = {"Authorization": f"Bearer {credentials.token}"}
    comments = []
    next_page_token = None

    while True:
        url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&maxResults=100"
        if next_page_token:
            url += f"&pageToken={next_page_token}"

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for item in data['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comment_id = item['snippet']['topLevelComment']['id']

                if is_english(comment) and not contains_link(comment):
                    comments.append((comment, comment_id))

            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                break
        else:
            st.error("Failed to fetch comments from YouTube API")
            break

    return comments


def is_english(comment):
    try:
        return detect(comment) == 'en'
    except:
        return False


def contains_link(comment):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.search(url_pattern, comment) is not None


def train_comment_classifier():
    df = pd.read_csv(TRAIN_DATA_FILE)
    X = df['Comment']
    y = df['Label']

    vectorizer = CountVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    classifier = MultinomialNB()
    classifier.fit(X_vec, y)
    return classifier, vectorizer


def predict_comment_category(classifier, vectorizer, comment):
    comment_vec = vectorizer.transform([comment])
    prediction = classifier.predict(comment_vec)
    return prediction[0]


def authenticate_youtube_api(credentials):
    youtube = build('youtube', 'v3', credentials=credentials)
    return youtube


def post_reply_to_comment(youtube, comment_id, reply_text):
    request_body = {
        'snippet': {
            'parentId': comment_id,
            'textOriginal': reply_text
        }
    }

    response = youtube.comments().insert(
        part='snippet',
        body=request_body
    ).execute()

    return response


def auto_reply_to_comments(youtube, comments, category):
    predefined_replies = {
        "Appreciation": "Thank you for your kind words! We appreciate your support!",
        "Suggestion": "Thanks for your suggestion! We will definitely consider it!"
    }

    for comment, comment_id in comments:
        reply_text = predefined_replies.get(category)
        if reply_text:
            try:
                post_reply_to_comment(youtube, comment_id, reply_text)
                st.success(f"Auto-reply posted to comment: {comment}")
            except Exception as e:
                st.error(f"Failed to post auto-reply: {e}")


st.title("Welcome To YTCM")

# Check if user is already logged in
credentials = load_credentials()

if credentials and credentials.expired and credentials.refresh_token:
    credentials.refresh(Request())

if st.experimental_get_query_params().get("reload"):
    st.experimental_set_query_params()
    st.experimental_rerun()

if credentials:
    st.success("You are logged in!")

    if st.button("Logout"):
        os.remove(TOKEN_FILE)
        st.success("You have logged out successfully!")
        st.experimental_set_query_params(reload="true")

    youtube = authenticate_youtube_api(credentials)

    video_url = st.text_input("Enter the YouTube video URL:")
    if video_url:
        video_id = video_url.split("v=")[-1]
        comments = fetch_youtube_data(credentials, video_id)

        if comments:
            classifier, vectorizer = train_comment_classifier()

            categories = {"Question": [], "Suggestion": [], "Appreciation": []}
            for comment, comment_id in comments:
                category = predict_comment_category(classifier, vectorizer, comment)
                categories[category].append((comment, comment_id))

            selected_category = st.selectbox("Select category to view comments:", categories.keys())

            st.write(f"### {selected_category} ({len(categories[selected_category])} comments)")

            for idx, (comment, comment_id) in enumerate(categories[selected_category]):
                st.write(f"- {comment}")

            if selected_category in ["Appreciation", "Suggestion"]:
                if st.button(f"Auto-reply to all {selected_category} comments"):
                    auto_reply_to_comments(youtube, categories[selected_category], selected_category)

            if selected_category == "Question":
                for idx, (comment, comment_id) in enumerate(categories["Question"]):
                    reply_key = f"reply_{idx}"
                    reply = st.text_input(f"Reply to question: {comment}", key=reply_key)
                    if reply:
                        st.write(f"**Your reply:** {reply}")
                        if st.button(f"Post reply to: {comment}"):
                            try:
                                post_reply_to_comment(youtube, comment_id, reply)
                                st.success("Your reply was posted!")
                            except Exception as e:
                                st.error(f"Failed to post your reply: {e}")
        else:
            st.write("No comments found or unable to fetch comments.")
else:
    authenticate_user()
