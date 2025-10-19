"""
BOOK RECOMMENDATION SYSTEM - AI-POWERED BOOK DISCOVERY
======================================================
Find your next favorite book using advanced machine learning algorithms.
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Book Recommender - AI Powered",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with DataCamp-inspired Theme
st.markdown("""
<style>
    /* Main background - Clean white/light gray like DataCamp */
    .stApp {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }

    /* Header styling - DataCamp green */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #05192d 0%, #1e3a5f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Lato', sans-serif;
    }

    .sub-header {
        font-size: 1.3rem;
        color: #5a6c7d;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    /* Section headers - DataCamp navy blue */
    h2, h3 {
        color: #05192d !important;
        font-weight: 700 !important;
    }

    /* Input fields - Clean DataCamp style */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #05192d !important;
        border: 2px solid #e0e5ea !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        font-size: 16px !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput input::placeholder {
        color: #9ba7b4 !important;
    }

    .stTextInput input:focus {
        border-color: #03ef62 !important;
        box-shadow: 0 0 0 3px rgba(3, 239, 98, 0.1) !important;
        outline: none !important;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 2px solid #e0e5ea !important;
        border-radius: 8px !important;
        color: #05192d !important;
    }

    .stSelectbox > div > div:hover {
        border-color: #03ef62 !important;
    }

    /* Radio buttons - DataCamp style */
    .stRadio > div {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #e0e5ea;
    }

    .stRadio label {
        color: #05192d !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }

    /* Slider - DataCamp green */
    .stSlider > div > div > div {
        background-color: #03ef62 !important;
    }

    /* Primary button - DataCamp signature green */
    .stButton > button {
        background: linear-gradient(135deg, #03ef62 0%, #00d45a 100%) !important;
        color: #05192d !important;
        border: none !important;
        padding: 14px 32px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(3, 239, 98, 0.25) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #00d45a 0%, #00bd51 100%) !important;
        box-shadow: 0 6px 16px rgba(3, 239, 98, 0.35) !important;
        transform: translateY(-2px) !important;
    }

    /* Book cards - Clean card design */
    .book-card {
        background: #ffffff;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #e0e5ea;
        box-shadow: 0 2px 8px rgba(5, 25, 45, 0.08);
        transition: all 0.3s ease;
    }

    .book-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(3, 239, 98, 0.15);
        border-color: #03ef62;
    }

    /* Selected book highlight */
    .selected-book {
        background: linear-gradient(135deg, rgba(3, 239, 98, 0.05) 0%, rgba(0, 212, 90, 0.05) 100%);
        padding: 28px;
        border-radius: 12px;
        border: 2px solid #03ef62;
        box-shadow: 0 4px 16px rgba(3, 239, 98, 0.1);
        margin: 20px 0;
    }

    /* Text colors - DataCamp dark blue/gray */
    p, label, span {
        color: #5a6c7d !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #05192d !important;
    }

    /* Strong text */
    strong {
        color: #05192d !important;
    }

    /* Links - DataCamp green */
    a {
        color: #03ef62 !important;
        text-decoration: none !important;
    }

    a:hover {
        color: #00bd51 !important;
    }

    /* Expander - Clean style */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        border: 1px solid #e0e5ea !important;
        border-radius: 8px !important;
        color: #05192d !important;
        font-weight: 600 !important;
    }

    .streamlit-expanderHeader:hover {
        background-color: rgba(3, 239, 98, 0.05) !important;
        border-color: #03ef62 !important;
    }

    /* Separator */
    hr {
        border-color: #e0e5ea !important;
        opacity: 1;
        margin: 2rem 0;
    }

    /* Success/Info messages */
    .stAlert {
        background-color: rgba(3, 239, 98, 0.1) !important;
        border: 1px solid #03ef62 !important;
        border-radius: 8px !important;
        color: #05192d !important;
    }

    /* Warning messages */
    .stWarning {
        background-color: rgba(255, 193, 7, 0.1) !important;
        border: 1px solid #ffc107 !important;
    }

    /* Spinner - DataCamp green */
    .stSpinner > div {
        border-top-color: #03ef62 !important;
    }

    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f8f9fa;
    }

    ::-webkit-scrollbar-thumb {
        background: #03ef62;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #00bd51;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load datasets with caching."""
    try:
        books = pd.read_csv('Books.csv', encoding='latin-1', low_memory=False)
        ratings = pd.read_csv('Ratings.csv', encoding='latin-1', low_memory=False)
        return books, ratings
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None


def get_similar_books_knn(book_title, ratings_df, books_df, n=10):
    """Get similar books using item-based collaborative filtering."""
    user_counts = ratings_df['User-ID'].value_counts()
    active_users = user_counts[user_counts >= 50].index
    filtered_ratings = ratings_df[ratings_df['User-ID'].isin(active_users)]

    book_counts = filtered_ratings['ISBN'].value_counts()
    popular_books = book_counts[book_counts >= 20].index
    filtered_ratings = filtered_ratings[filtered_ratings['ISBN'].isin(popular_books)]

    try:
        book_pivot = filtered_ratings.pivot_table(
            index='ISBN',
            columns='User-ID',
            values='Book-Rating'
        ).fillna(0)

        book_isbn = books_df[books_df['Book-Title'] == book_title]['ISBN'].values
        if len(book_isbn) == 0:
            return pd.DataFrame()

        book_isbn = book_isbn[0]

        if book_isbn not in book_pivot.index:
            return pd.DataFrame()

        book_vector = book_pivot.loc[book_isbn].values.reshape(1, -1)
        similarities = cosine_similarity(book_vector, book_pivot.values)[0]

        similar_indices = similarities.argsort()[::-1][1:n+1]
        similar_isbns = book_pivot.index[similar_indices]
        similar_scores = similarities[similar_indices]

        recommendations = books_df[books_df['ISBN'].isin(similar_isbns)].copy()
        recommendations['Similarity'] = recommendations['ISBN'].map(dict(zip(similar_isbns, similar_scores)))
        recommendations = recommendations.sort_values('Similarity', ascending=False)

        return recommendations.drop_duplicates('ISBN').head(n)
    except Exception as e:
        return pd.DataFrame()


def get_content_based_recommendations(book_title, books_df, ratings_df, n=10):
    """Get recommendations based on book content (author, publisher)."""
    book_counts = ratings_df['ISBN'].value_counts()
    popular_books = book_counts[book_counts >= 10].index

    filtered_books = books_df[books_df['ISBN'].isin(popular_books)].copy()
    filtered_books = filtered_books.drop_duplicates('Book-Title')

    filtered_books['combined_features'] = (
        filtered_books['Book-Author'].fillna('') + ' ' +
        filtered_books['Publisher'].fillna('')
    )

    try:
        vectorizer = CountVectorizer(stop_words='english')
        feature_vectors = vectorizer.fit_transform(filtered_books['combined_features'])

        book_idx = filtered_books[filtered_books['Book-Title'] == book_title].index
        if len(book_idx) == 0:
            return pd.DataFrame()

        book_idx = book_idx[0]

        similarities = cosine_similarity(
            feature_vectors[book_idx:book_idx+1],
            feature_vectors
        )[0]

        similar_indices = similarities.argsort()[::-1][1:n+1]
        recommendations = filtered_books.iloc[similar_indices].copy()
        recommendations['Similarity'] = similarities[similar_indices]

        avg_ratings = ratings_df.groupby('ISBN')['Book-Rating'].mean()
        recommendations['AvgRating'] = recommendations['ISBN'].map(avg_ratings)

        return recommendations.head(n)
    except Exception as e:
        return pd.DataFrame()


def main():
    """Main application."""

    # Header
    st.markdown('<h1 class="main-header">📚 BOOK RECOMMENDATION SYSTEM</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover your next favorite book with machine learning algorithms</p>', unsafe_allow_html=True)

    # Load data
    books_df, ratings_df = load_data()

    if books_df is None or ratings_df is None:
        st.error("❌ Failed to load data files. Please ensure Books.csv and Ratings.csv are in the same directory.")
        return

    # Get books with sufficient ratings
    book_counts = ratings_df['ISBN'].value_counts()
    popular_books_isbn = book_counts[book_counts >= 20].index
    available_books = books_df[books_df['ISBN'].isin(popular_books_isbn)]['Book-Title'].unique()

    # Main content
    st.markdown("---")

    # Search section
    st.markdown("## 🔍 Search for a Book")

    search_query = st.text_input(
        "",
        placeholder="Type book title (e.g., Harry Potter, The Da Vinci Code, 1984)...",
        label_visibility="collapsed"
    )

    # Filter and select book
    if search_query:
        filtered_books = [book for book in available_books if search_query.lower() in book.lower()]
        if filtered_books:
            selected_book = st.selectbox(
                f"📖 Found {len(filtered_books)} books:",
                filtered_books[:50]
            )
        else:
            st.warning(f"⚠️ No books found matching '{search_query}'. Try a different search.")
            # Get popular books from ratings
            popular_df = ratings_df.groupby('ISBN').size().sort_values(ascending=False).head(20)
            popular_titles = books_df[books_df['ISBN'].isin(popular_df.index)]['Book-Title'].unique()[:20]
            selected_book = st.selectbox("📚 Try these popular books:", popular_titles)
    else:
        # Show popular books
        popular_df = ratings_df.groupby('ISBN').size().sort_values(ascending=False).head(20)
        popular_titles = books_df[books_df['ISBN'].isin(popular_df.index)]['Book-Title'].unique()[:20]
        selected_book = st.selectbox("📚 Or choose from popular books:", popular_titles)

    st.markdown("---")

    # Algorithm selection
    st.markdown("## ⚙️ Recommendation Settings")

    col1, col2 = st.columns([2, 1])

    with col1:
        algorithm = st.radio(
            "**Choose Algorithm:**",
            ['🤝 Collaborative Filtering', '📖 Content-Based Filtering'],
            help="Collaborative: Recommends based on user preferences | Content-Based: Recommends based on book metadata"
        )

    with col2:
        n_recommendations = st.slider("**Number of results:**", 3, 15, 5)

    st.markdown("---")

    # Get recommendations button
    if st.button("🎯 FIND SIMILAR BOOKS", type="primary"):

        # Show selected book
        st.markdown("## 📖 Your Selected Book")

        book_info = books_df[books_df['Book-Title'] == selected_book].iloc[0]

        col1, col2 = st.columns([1, 3])

        with col1:
            try:
                if pd.notna(book_info['Image-URL-M']) and book_info['Image-URL-M']:
                    st.image(book_info['Image-URL-M'], width=150)
                else:
                    st.markdown("### 📚")
            except Exception as e:
                st.markdown("### 📚")

        with col2:
            st.markdown(f"### {book_info['Book-Title']}")
            st.markdown(f"**✍️ Author:** {book_info['Book-Author']}")
            st.markdown(f"**🏢 Publisher:** {book_info.get('Publisher', 'N/A')}")

            book_isbn = book_info['ISBN']
            book_ratings = ratings_df[ratings_df['ISBN'] == book_isbn]['Book-Rating']
            if len(book_ratings) > 0:
                avg_rating = book_ratings[book_ratings > 0].mean()
                num_ratings = len(book_ratings)
                st.markdown(f"**⭐ Rating:** {avg_rating:.1f}/10 ({num_ratings:,} ratings)")

        st.markdown("---")

        # Get recommendations
        algo_clean = algorithm.split()[1]  # Remove emoji
        with st.spinner(f'🔍 Finding similar books using {algo_clean}...'):
            if 'Collaborative' in algorithm:
                recommendations = get_similar_books_knn(selected_book, ratings_df, books_df, n=n_recommendations)
            else:
                recommendations = get_content_based_recommendations(selected_book, books_df, ratings_df, n=n_recommendations)

        if recommendations.empty:
            st.warning("⚠️ Could not find similar books. Try a different book or algorithm.")
        else:
            st.markdown(f"## 🎉 Recommended Books for You")
            st.markdown(f"*Based on **{algo_clean}***")

            for idx, book in recommendations.iterrows():
                st.markdown('<div class="book-card">', unsafe_allow_html=True)

                col1, col2 = st.columns([1, 4])

                with col1:
                    try:
                        if pd.notna(book['Image-URL-M']) and book['Image-URL-M']:
                            st.image(book['Image-URL-M'], width=120)
                        else:
                            st.markdown("### 📚")
                    except Exception as e:
                        st.markdown("### 📚")

                with col2:
                    st.markdown(f"### {book['Book-Title']}")
                    st.markdown(f"**✍️ Author:** {book['Book-Author']}")
                    st.markdown(f"**🏢 Publisher:** {book.get('Publisher', 'N/A')}")

                    if 'Similarity' in book and not pd.isna(book['Similarity']):
                        similarity_pct = book['Similarity'] * 100
                        st.markdown(f"**🎯 Match Score:** {similarity_pct:.1f}%")

                    if 'AvgRating' in book and not pd.isna(book['AvgRating']):
                        st.markdown(f"**⭐ Rating:** {book['AvgRating']:.1f}/10")

                st.markdown('</div>', unsafe_allow_html=True)

    # Tips section
    with st.expander("💡 How does it work?"):
        st.markdown("""
        ### 🤝 Collaborative Filtering
        - Analyzes patterns from users with similar reading preferences
        - Finds books that readers with similar taste enjoyed
        - Best for discovering popular and highly-rated books

        ### 📖 Content-Based Filtering
        - Matches books based on author, publisher, and other metadata
        - Great for finding books with similar writing styles
        - Helps discover new authors in your favorite genres

        **💡 Tip:** Try both methods to get diverse recommendations!
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #5a6c7d; font-size: 14px;'>Powered by Collaborative & Content-Based Filtering | 271K Books, 1.1M Ratings</p>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
