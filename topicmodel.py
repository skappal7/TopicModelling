import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import MDS

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.write(df.head())

    # Perform topic modelling
    st.header("Topic Modelling")
    num_topics = st.slider("Select the number of topics", min_value=2, max_value=10, value=4)

    # Select text column
    text_column_name = st.selectbox("Select the column containing text data", df.columns)

    # Preprocess the text data
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(df[text_column_name].astype(str))

    # Train the LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    doc_topic_dist = lda.fit_transform(doc_term_matrix)

    # Display topic modelling results
    st.subheader("1. Topic Overview")
    st.write("Here are the main topics identified in your data. Each topic is represented by its most significant words.")
    
    feature_names = vectorizer.get_feature_names_out()
    topic_names = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-5 - 1:-1]]
        st.write(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
        topic_names[topic_idx] = st.text_input(f"Give a name to Topic {topic_idx + 1}", f"Topic {topic_idx + 1}")

    # Assign topics to each document
    df['Primary Topic'] = doc_topic_dist.argmax(axis=1)
    df['Primary Topic Name'] = df['Primary Topic'].map(topic_names)

    # Display document-topic assignment
    st.subheader("2. Document-Topic Assignment")
    st.write("Here's a sample of documents and their primary assigned topics. This shows you which topics are most relevant to each document.")
    st.write(df[[text_column_name, 'Primary Topic Name']].sample(min(10, len(df))))

    # Topic filtering
    selected_topic = st.selectbox("Filter documents by topic:", list(topic_names.values()))
    filtered_df = df[df['Primary Topic Name'] == selected_topic]
    st.write(f"Documents primarily about {selected_topic}:")
    st.write(filtered_df[[text_column_name, 'Primary Topic Name']].head())

    # Visualize the distribution of topics
    st.subheader("3. Topic Distribution")
    st.write("This chart shows how prevalent each topic is across all documents. Taller bars indicate topics that appear more frequently in the dataset.")
    topic_distribution = df['Primary Topic Name'].value_counts()
    fig = px.bar(x=topic_distribution.index, y=topic_distribution.values,
                 labels={'x': 'Topic', 'y': 'Number of Documents'},
                 title='Distribution of Topics')
    st.plotly_chart(fig)

    # Interactive topic visualization
    st.subheader("4. Topic Similarity Visualization")
    st.write("This plot shows how similar topics are to each other. Topics that are closer together in the plot are more similar in content.")
    
    # MDS projection
    mds = MDS(n_components=2, random_state=42)
    topic_coordinates = mds.fit_transform(lda.components_)

    # Create scatter plot for topics
    fig = px.scatter(
        x=topic_coordinates[:, 0], y=topic_coordinates[:, 1],
        text=[topic_names[i] for i in range(num_topics)],
        title='Topic Similarity Visualization'
    )
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig)

    # Display top terms for selected topic
    st.subheader("5. Word Importance in Topics")
    st.write("This shows the most important words for a selected topic. The longer the bar, the more important the word is in defining the topic.")
    selected_topic_idx = st.selectbox("Select a topic to see top terms:", range(num_topics), format_func=lambda x: topic_names[x])
    top_terms = [feature_names[i] for i in lda.components_[selected_topic_idx].argsort()[:-10 - 1:-1]]
    term_strengths = lda.components_[selected_topic_idx][lda.components_[selected_topic_idx].argsort()[:-10 - 1:-1]]

    term_fig = px.bar(x=top_terms, y=term_strengths, 
                      labels={'x': 'Term', 'y': 'Importance'},
                      title=f'Top 10 Terms in {topic_names[selected_topic_idx]}')
    st.plotly_chart(term_fig)

    st.write(f"Interpretation: These words are the most characteristic of {topic_names[selected_topic_idx]}. They give you an idea of what this topic is about and what kind of documents it might include.")

st.write("This topic modeling tool helps you understand the main themes in your text data, how they relate to each other, and how they're distributed across your documents. Use it to gain insights into the content and structure of your text corpus.")
