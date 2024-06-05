

import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import streamlit as st


def read_article(text):
    sentences = text.split(". ")
    processed_sentences = []
    for sentence in sentences:
        cleaned_sentence = sentence.replace("[^a-zA-Z]", " ").split()
        processed_sentences.append(cleaned_sentence)
    if processed_sentences[-1] == ['']:
        processed_sentences.pop()
    return processed_sentences



def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)




def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix



def gen_summary(text, top_n=5):
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    sentences = read_article(text)
    sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    summarize = []
    for i in range(min(top_n, len(ranked_sentences))):
        summarize.append(" ".join(ranked_sentences[i][1]))

    return ". ".join(summarize)


st.title("Text Summarizer")

option = st.radio("Choose input method:", ('Write Text', 'Upload File'))

if option == 'Write Text':
    text_input = st.text_area("Enter your text here", height=200)
    summary_length = st.slider("Select number of sentences for summary", 1, 10, 5)
    
    if st.button("Generate Summary"):
        if text_input.strip() != "":
            summary = gen_summary(text_input, summary_length)
            st.write("Summary:")
            st.write(summary)
        else:
            st.write("Please enter some text to summarize.")
            
elif option == 'Upload File':
    uploaded_file = st.file_uploader("Choose a text file", type="txt")
    summary_length = st.slider("Select number of sentences for summary", 1, 10, 5)
    
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        st.write("Original Text:")
        st.write(file_content)
        
        if st.button("Generate Summary"):
            summary = gen_summary(file_content, summary_length)
            st.write("Summary:")
            st.write(summary)
