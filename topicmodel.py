import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Logo embedded in the source code
logo_url = "https://humach.com/wp-content/uploads/2023/01/HuMach_logo-bold.png"  # Replace with your logo URL or embed the logo directly

# Set page title and icon
st.set_page_config(page_title="Text Analyser", page_icon=":pencil:")

# Set app name and subheading
st.title("Text Analyser")
st.subheader("Developer: Sunil Kappal")

# Display logo at the top right-hand side
st.image(logo_url, width=100, use_column_width=False, output_format='auto')

# Upload file for analysis
uploaded_file = st.file_uploader("Upload CSV or Text file for analysis", type=["csv", "txt"])

if uploaded_file is not None:
    # Read the file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        df = pd.DataFrame({'Text': uploaded_file.readlines()})
    else:
        st.error("Invalid file format. Please upload a CSV or Text file.")
        st.stop()

    # Display uploaded data
    st.write("Uploaded Data:")
    st.dataframe(df)

    # Perform topic modelling
    st.header("Topic Modelling:")
    num_topics = st.slider("Select the number of topics", min_value=2, max_value=10, value=4)

    # Topic Modelling using scikit-learn's LatentDirichletAllocation
    text_column_name = st.selectbox("Select the column containing text data", df.columns)

    # Fit CountVectorizer to the text data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df[text_column_name])

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    # Display topic modelling results
    for topic_idx, topic in enumerate(lda.components_):
        st.write(f"Topic {topic_idx + 1}:")
        top_words_idx = topic.argsort()[-5:][::-1]
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
        st.write(", ".join(top_words))

    # Assign topics to each document
    df['Topic'] = lda.transform(X).argmax(axis=1)

    # Display 'Text' and 'Topic' columns if they exist
    if 'Text' in df.columns and 'Topic' in df.columns:
        st.write("Assigned Topics:")
        st.dataframe(df[['Text', 'Topic']])

        # Visualize the distribution of topics
        st.header("Topic Distribution:")
        topic_distribution = df['Topic'].value_counts()
        st.bar_chart(topic_distribution)
    else:
        st.warning("Please check your data processing steps. 'Text' and 'Topic' columns not found in the DataFrame.")
