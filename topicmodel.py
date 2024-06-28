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
    st.write(df.head())

    # Perform topic modelling
    st.header("Topic Modelling:")
    num_topics = st.slider("Select the number of topics", min_value=2, max_value=10, value=4)

    # Select text column
    text_column_name = st.selectbox("Select the column containing text data", df.columns)

    # Preprocess the text data
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(df[text_column_name].astype(str))

    # Train the LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    # Display topic modelling results
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-5 - 1:-1]]
        st.write(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    # Assign topics to each document
    doc_topics = lda.transform(doc_term_matrix)
    df['Topic'] = doc_topics.argmax(axis=1)

    # Display 'Text' and 'Topic' columns
    st.write("Assigned Topics:")
    st.dataframe(df[[text_column_name, 'Topic']])

    # Visualize the distribution of topics
    st.header("Topic Distribution:")
    topic_distribution = df['Topic'].value_counts().sort_index()
    fig = px.bar(x=topic_distribution.index, y=topic_distribution.values,
                 labels={'x': 'Topic', 'y': 'Count'},
                 title='Distribution of Topics')
    st.plotly_chart(fig)

    # Interactive topic visualization
    st.header("Interactive Topic Visualization:")
    
    # Prepare data for visualization
    topic_term_dists = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    doc_topic_dists = doc_topics / doc_topics.sum(axis=1)[:, np.newaxis]
    doc_lengths = doc_term_matrix.sum(axis=1).A1
    vocab = vectorizer.get_feature_names_out()
    term_frequency = doc_term_matrix.sum(axis=0).A1

    # MDS projection
    mds = MDS(n_components=2, random_state=42)
    topic_coordinates = mds.fit_transform(topic_term_dists)

    # Create scatter plot for topics
    trace1 = go.Scatter(
        x=topic_coordinates[:, 0],
        y=topic_coordinates[:, 1],
        mode='markers',
        marker=dict(
            size=15, 
            color=[i for i in range(num_topics)],
            colorscale='Viridis', 
            showscale=True
        ),
        text=[f'Topic {i+1}' for i in range(num_topics)],
        hoverinfo='text'
    )

    layout = go.Layout(
        title='Topic Visualization',
        xaxis=dict(title='First Dimension'),
        yaxis=dict(title='Second Dimension'),
        showlegend=False
    )

    fig = go.Figure(data=[trace1], layout=layout)
    st.plotly_chart(fig)

    # Display top terms for selected topic
    selected_topic = st.selectbox("Select a topic to see top terms:", range(num_topics))
    top_terms = [vocab[i] for i in topic_term_dists[selected_topic].argsort()[:-10 - 1:-1]]
    term_strengths = topic_term_dists[selected_topic][topic_term_dists[selected_topic].argsort()[:-10 - 1:-1]]

    term_fig = px.bar(x=top_terms, y=term_strengths, 
                      labels={'x': 'Term', 'y': 'Strength'},
                      title=f'Top 10 Terms in Topic {selected_topic + 1}')
    st.plotly_chart(term_fig)
