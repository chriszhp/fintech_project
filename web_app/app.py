import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
from jieba import analyse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
import base64
from torch import device, cuda
from transformers import pipeline
import time

# Set up the Streamlit app
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

device = device(0 if cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    model1 = pipeline("zero-shot-classification", model="./src/model/classification", tokenizer="./src/model/classification", device=device)
    model2 = pipeline("sentiment-analysis", model="./src/model/sentiment", tokenizer="./src/model/sentiment", device=device)
    return model1, model2

model1, model2 = load_models()

# Display the app title
st.title("Sentiment Analysis")

def preprocess_sentences(sentences, stopwords):
    words = []
    for sentence in sentences:
        sentence = str(sentence.strip())
        if any(ord(c) > 128 for c in sentence):  # Chinese sentence
            tokens = jieba.cut(sentence, cut_all=False)
        else:  # English sentence
            tokens = word_tokenize(sentence.lower())
        words.extend([word for word in tokens if word not in stopwords])

    return words


def generate_wordcloud(comments):
    stopwords_chinese_file = './src/stopwords/chinese_stopwords.txt'
    analyse.set_stop_words(stopwords_chinese_file)
    with open(stopwords_chinese_file, 'r', encoding='utf-8') as f:
        stopwords_chinese = set([line.strip() for line in f.readlines()])
    stopwords_english = set(nltk_stopwords.words('english'))
    stopwords = stopwords_chinese.union(stopwords_english)
    font_path = './src/font/SourceHanSerif-VF.ttf.ttc'
    words = preprocess_sentences(comments, stopwords)
    word_freq = Counter(words)
    w = WordCloud(scale=2, font_path=font_path, max_font_size=100, background_color='black', colormap='Blues', stopwords=stopwords)
    w.generate_from_frequencies(word_freq)
    return w

# Create a sidebar for user inputs
with st.sidebar:
    st.header("Input")
    comment = st.text_area("Comment", value="ELI係合約,係幫對手打底,並不是真實持有正股,除非你被行使,不過被行使接貨就多數會凶多吉少", height=100)
    keywords = st.text_input("Keywords (comma-separated)", value="price,risk,channel,promotion,other")
    process_button = st.button("Process")

# Process the comment when the button is clicked
if process_button:
    category_scores = model1(comment, keywords.split(','))
    sentiment_score = model2(comment)
    st.header("Results")
    sentiment_label = sentiment_score[0]['label']
    if sentiment_label == 'positive':
        sentiment_color = 'green'
    elif sentiment_label == 'negative':
        sentiment_color = 'red'
    else:
        sentiment_color = 'white'
    st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{sentiment_score[0]['label']}</span> | "
                f"**Sentiment score:** {sentiment_score[0]['score']:.6f}", unsafe_allow_html=True)
    st.write("Category scores:")
    st.write(pd.DataFrame({'Category': category_scores['labels'], 'Score': category_scores['scores']}))

# Create a file uploader for batch processing
uploaded_file = st.sidebar.file_uploader("Upload a CSV file for batch processing")

# Process the uploaded file
if uploaded_file is not None:
    comments_df = pd.read_csv(uploaded_file,header=None)
    if st.sidebar.button("Batch Process"):
        results = []
        progress_bar = st.progress(0)
        start_time = time.time()  # Start the timer
        for _, row in comments_df.iterrows():
            comment = str(row[0].strip())
            if not comment:
                continue

            category_scores = model1(comment, keywords.split(','))
            sentiment_score = model2(comment)

            results.append({
                'comment': comment,
                'category': category_scores['labels'][0],
                'category_scores': category_scores['scores'][0],
                'sentiment': sentiment_score[0]['label'],
                'sentiment_score': sentiment_score[0]['score']
            })
            progress_bar.progress((_ + 1) / len(comments_df))
        results_df = pd.DataFrame(results)
        end_time = time.time()  # Stop the timer
        st.header("Batch Results")
        st.write(f"Batch processing took {end_time - start_time:.2f} seconds")
        st.write(results_df)

        # Add an option to download the results as a CSV file
        def to_csv_download_link(df, filename="results.csv", link_name="Download as CSV"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_name}</a>'

        st.markdown(to_csv_download_link(results_df), unsafe_allow_html=True)

    # Add an option to generate a wordcloud
    if st.sidebar.button("Generate Wordcloud"):
        wc = generate_wordcloud(comments_df.iloc[:, 0].tolist())
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
