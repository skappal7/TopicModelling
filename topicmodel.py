import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models
import tempfile
import base64

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # Perform topic modelling
    st.header("Topic Modelling:")
    num_topics = st.slider("Select the number of topics", min_value=2, max_value=10, value=4)

    # Topic Modelling using Gensim's LatentDirichletAllocation
    text_column_name = st.selectbox("Select the column containing text data", df.columns)

    # Preprocess the text data
    texts = df[text_column_name].astype(str).apply(lambda x: x.split())

    # Create a dictionary and corpus for Gensim
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train the LDA model using Gensim
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=42)

    # Display topic modelling results
    for topic_idx in range(num_topics):
        st.write(f"Topic {topic_idx + 1}:")
        st.write(", ".join([word for word, prob in lda.show_topic(topic_idx, topn=5)]))

    # Assign topics to each document
    doc_topics = [max(doc, key=lambda x: x[1])[0] for doc in lda[corpus]]
    df['Topic'] = doc_topics

    # Display 'Text' and 'Topic' columns if they exist
    if text_column_name in df.columns and 'Topic' in df.columns:
        st.write("Assigned Topics:")
        st.dataframe(df[[text_column_name, 'Topic']])

        # Visualize the distribution of topics
        st.header("Topic Distribution:")
        topic_distribution = df['Topic'].value_counts()
        st.bar_chart(topic_distribution)

        # PyLDAvis visualization
        st.header("PyLDAvis Visualization:")

        # Prepare the data for PyLDAvis
        vis_data = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)

        # Save the visualization to a temporary HTML file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
            pyLDAvis.save_html(vis_data, tmpfile.name)
            
            # Read the HTML file and encode it
            with open(tmpfile.name, 'rb') as f:
                html_bytes = f.read()
            encoded = base64.b64encode(html_bytes).decode()
            
        # Display the visualization in an iframe
        st.components.v1.html(f'<iframe src="data:text/html;base64,{encoded}" width="100%" height="800px"></iframe>', height=800)

    else:
        st.warning("Please check your data processing steps. 'Text' and 'Topic' columns not found in the DataFrame.")
