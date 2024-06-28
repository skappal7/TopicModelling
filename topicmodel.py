import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import tempfile
import base64

# ... (previous code remains the same) ...

if uploaded_file is not None:
    # ... (previous code remains the same) ...

    # Perform topic modelling
    st.header("Topic Modelling:")
    num_topics = st.slider("Select the number of topics", min_value=2, max_value=10, value=4)

    # Topic Modelling using scikit-learn's LatentDirichletAllocation
    text_column_name = st.selectbox("Select the column containing text data", df.columns)

    # Fit CountVectorizer to the text data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df[text_column_name])

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_output = lda.fit_transform(X)

    # Display topic modelling results
    for topic_idx, topic in enumerate(lda.components_):
        st.write(f"Topic {topic_idx + 1}:")
        top_words_idx = topic.argsort()[-5:][::-1]
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
        st.write(", ".join(top_words))

    # Assign topics to each document
    df['Topic'] = lda_output.argmax(axis=1)

    # Display 'Text' and 'Topic' columns if they exist
    if 'Text' in df.columns and 'Topic' in df.columns:
        st.write("Assigned Topics:")
        st.dataframe(df[['Text', 'Topic']])

        # Visualize the distribution of topics
        st.header("Topic Distribution:")
        topic_distribution = df['Topic'].value_counts()
        st.bar_chart(topic_distribution)

        # PyLDAvis visualization
        st.header("PyLDAvis Visualization:")
        
        # Prepare the visualization
        feature_names = vectorizer.get_feature_names_out()
        doc_lengths = X.sum(axis=1).getA1()
        vocab = feature_names
        term_frequency = X.sum(axis=0).getA1()

        # Create and save the visualization
        data = pyLDAvis.prepare(
            topic_word_distrib=lda.components_,
            doc_topic_distrib=lda_output,
            doc_lengths=doc_lengths,
            vocab=vocab,
            term_frequency=term_frequency,
            sort_topics=False
        )
        
        # Save the visualization to a temporary HTML file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
            pyLDAvis.save_html(data, tmpfile.name)
            
            # Read the HTML file and encode it
            with open(tmpfile.name, 'rb') as f:
                html_bytes = f.read()
            encoded = base64.b64encode(html_bytes).decode()
            
        # Display the visualization in an iframe
        st.components.v1.html(f'<iframe src="data:text/html;base64,{encoded}" width="100%" height="800px"></iframe>', height=800)

    else:
        st.warning("Please check your data processing steps. 'Text' and 'Topic' columns not found in the DataFrame.")
