# %%
# ====================
# BOOK RECOMMENDATION SYSTEM
# ====================

# This comprehensive book recommendation system demonstrates expertise in recommendation algorithms,
# data analysis, and machine learning modeling. The project showcases the implementation of multiple
# recommendation approaches: popularity-based, collaborative filtering, and content-based filtering.

# %%
# ====================
# 1. IMPORTS & SETUP
# ====================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import re
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# %%
# ====================
# 2. DATA LOADING
# ====================

# Load datasets
books = pd.read_csv('Books.csv', encoding='latin-1', low_memory=False)
ratings = pd.read_csv('Ratings.csv', encoding='latin-1', low_memory=False)
users = pd.read_csv('Users.csv', encoding='latin-1', low_memory=False)

print("Dataset Shapes:")
print(f"Books: {books.shape}")
print(f"Ratings: {ratings.shape}")
print(f"Users: {users.shape}")

# %%
# ====================
# 3. DATA PREPROCESSING
# ====================

# Merge datasets
books_data = books.merge(ratings, on="ISBN")
df = books_data.copy()

# Preprocessing
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop(columns=["ISBN", "Year-Of-Publication", "Image-URL-S", "Image-URL-M"], axis=1, inplace=True)
df.drop(index=df[df["Book-Rating"] == 0].index, inplace=True)
df["Book-Title"] = df["Book-Title"].apply(lambda x: re.sub(r"[\W_]+", " ", x).strip())

print(f"After preprocessing: {df.shape}")

# %%
# ====================
# 4. POPULARITY-BASED RECOMMENDATION SYSTEM
# ====================

def popular_books(df, n=100):
    rating_count = df.groupby("Book-Title").count()["Book-Rating"].reset_index()
    rating_count.rename(columns={"Book-Rating": "NumberOfVotes"}, inplace=True)

    rating_average = df.groupby("Book-Title")["Book-Rating"].mean().reset_index()
    rating_average.rename(columns={"Book-Rating": "AverageRatings"}, inplace=True)

    popularBooks = rating_count.merge(rating_average, on="Book-Title")

    C = popularBooks["AverageRatings"].mean()
    m = popularBooks["NumberOfVotes"].quantile(0.90)

    popularBooks = popularBooks[popularBooks["NumberOfVotes"] >= 250]
    popularBooks["Popularity"] = popularBooks.apply(
        lambda x: ((x["NumberOfVotes"] * x["AverageRatings"]) + (m * C)) / (x["NumberOfVotes"] + m), axis=1
    )

    popularBooks = popularBooks.sort_values(by="Popularity", ascending=False)
    return popularBooks[["Book-Title", "NumberOfVotes", "AverageRatings", "Popularity"]].reset_index(drop=True).head(n)

# Get top 10 popular books
top_books = popular_books(df, 10)
print("Top 10 Popular Books:")
print(top_books)

# %%
# ====================
# 5. ITEM-BASED COLLABORATIVE FILTERING
# ====================

def item_based(bookTitle):
    bookTitle = str(bookTitle)

    if bookTitle in df["Book-Title"].values:
        rating_count = pd.DataFrame(df["Book-Title"].value_counts())
        rating_count.columns = ["count"]  # Rename the column
        rare_books = rating_count[rating_count["count"] <= 200].index
        common_books = df[~df["Book-Title"].isin(rare_books)]

        if bookTitle in rare_books:
            most_common = pd.Series(common_books["Book-Title"].unique()).sample(3).values
            print("No Recommendations for this Book ☹️ \n ")
            print("YOU MAY TRY: \n ")
            print(f"{most_common[0]}", "\n")
            print(f"{most_common[1]}", "\n")
            print(f"{most_common[2]}", "\n")
        else:
            common_books_pivot = common_books.pivot_table(index=["User-ID"], columns=["Book-Title"], values="Book-Rating")
            title = common_books_pivot[bookTitle]
            recommendation_df = pd.DataFrame(common_books_pivot.corrwith(title).sort_values(ascending=False)).reset_index(drop=False)

            if bookTitle in [title for title in recommendation_df["Book-Title"]]:
                recommendation_df = recommendation_df.drop(recommendation_df[recommendation_df["Book-Title"] == bookTitle].index[0])

            less_rating = []
            for i in recommendation_df["Book-Title"]:
                if df[df["Book-Title"] == i]["Book-Rating"].mean() < 5:
                    less_rating.append(i)

            if recommendation_df.shape[0] - len(less_rating) > 5:
                recommendation_df = recommendation_df[~recommendation_df["Book-Title"].isin(less_rating)]

            recommendation_df = recommendation_df[0:5]
            recommendation_df.columns = ["Book-Title", "Correlation"]

            print(f"Recommendations for '{bookTitle}':")
            for i, row in recommendation_df.iterrows():
                print(f"{i+1}. {row['Book-Title']} (Correlation: {row['Correlation']:.3f})")

    else:
        print("❌ COULD NOT FIND ❌")

# Test item-based recommendations
item_based("The Da Vinci Code")
item_based("Harry Potter and the Chamber of Secrets (Book 2)")

# %%
# ====================
# 6. USER-BASED COLLABORATIVE FILTERING
# ====================

# Filter users who have rated more than 200 books
new_df = df[df['User-ID'].map(df['User-ID'].value_counts()) > 200]
users_pivot = new_df.pivot_table(index=["User-ID"], columns=["Book-Title"], values="Book-Rating")
users_pivot.fillna(0, inplace=True)

def users_choice(id):
    users_fav = new_df[new_df["User-ID"] == id].sort_values(["Book-Rating"], ascending=False)[0:5]
    return users_fav

def user_based(new_df, id):
    if id not in new_df["User-ID"].values:
        print("❌ User NOT FOUND ❌")
        return []

    index = np.where(users_pivot.index == id)[0][0]
    similarity = cosine_similarity(users_pivot)
    similar_users = list(enumerate(similarity[index]))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[0:5]

    user_rec = []
    for i in similar_users:
        data = df[df["User-ID"] == users_pivot.index[i[0]]]
        user_rec.extend(list(data.drop_duplicates("User-ID")["User-ID"].values))

    return user_rec[:5]

def common(new_df, user, user_id):
    x = new_df[new_df["User-ID"] == user_id]
    recommend_books = []
    user = list(user)

    for i in user:
        y = new_df[(new_df["User-ID"] == i)]
        books = y.loc[~y["Book-Title"].isin(x["Book-Title"]), :]
        books = books.sort_values(["Book-Rating"], ascending=False)[0:5]
        recommend_books.extend(books["Book-Title"].values)

    return recommend_books[0:5]

# Test user-based recommendations
user_id = new_df["User-ID"].value_counts().index[0]  # Most active user
user_choice_df = users_choice(user_id)
user_based_rec = user_based(new_df, user_id)
books_for_user = common(new_df, user_based_rec, user_id)

print(f"\nUser {user_id} favorites:")
for i, row in user_choice_df.iterrows():
    print(f"{i+1}. {row['Book-Title']} (Rating: {row['Book-Rating']})")

print(f"\nRecommended books for User {user_id}:")
for i, book in enumerate(books_for_user, 1):
    print(f"{i}. {book}")

# %%
# ====================
# 7. CONTENT-BASED FILTERING
# ====================

def content_based(bookTitle):
    bookTitle = str(bookTitle)

    if bookTitle in df["Book-Title"].values:
        rating_count = pd.DataFrame(df["Book-Title"].value_counts())
        rating_count.columns = ["count"]  # Rename the column
        rare_books = rating_count[rating_count["count"] <= 200].index
        common_books = df[~df["Book-Title"].isin(rare_books)]

        if bookTitle in rare_books:
            most_common = pd.Series(common_books["Book-Title"].unique()).sample(3).values
            print("No Recommendations for this Book ☹️ \n ")
            print("YOU MAY TRY: \n ")
            print(f"{most_common[0]}", "\n")
            print(f"{most_common[1]}", "\n")
            print(f"{most_common[2]}", "\n")
        else:
            common_books = common_books.drop_duplicates(subset=["Book-Title"])
            common_books.reset_index(inplace=True)
            common_books["index"] = [i for i in range(common_books.shape[0])]
            targets = ["Book-Title", "Book-Author", "Publisher"]
            common_books["all_features"] = [" ".join(common_books[targets].iloc[i,].values) for i in range(common_books[targets].shape[0])]

            vectorizer = CountVectorizer()
            common_booksVector = vectorizer.fit_transform(common_books["all_features"])
            similarity = cosine_similarity(common_booksVector)

            index = common_books[common_books["Book-Title"] == bookTitle]["index"].values[0]
            similar_books = list(enumerate(similarity[index]))
            similar_booksSorted = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:6]

            books = []
            for i in range(len(similar_booksSorted)):
                books.append(common_books[common_books["index"] == similar_booksSorted[i][0]]["Book-Title"].item())

            print(f"Content-based recommendations for '{bookTitle}':")
            for i, book in enumerate(books, 1):
                rating = df[df["Book-Title"] == book]["Book-Rating"].mean()
                print(f"{i}. {book} (Avg Rating: {rating:.1f})")

    else:
        print("❌ COULD NOT FIND ❌")

# Test content-based recommendations
content_based("The Da Vinci Code")
content_based("Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))")

# %%
# ====================
# 8. MODEL EVALUATION & RESULTS
# ====================

print("\n" + "="*60)
print("BOOK RECOMMENDATION SYSTEM - SUMMARY")
print("="*60)

print("\n📊 Dataset Statistics:")
print(f"• Total books: {books.shape[0]:,}")
print(f"• Total ratings: {ratings.shape[0]:,}")
print(f"• Total users: {users.shape[0]:,}")
print(f"• Active users (rated >200 books): {len(new_df['User-ID'].unique()):,}")

print("\n🤖 Implemented Models:")
print("• Popularity-based: Recommends trending books")
print("• Item-based Collaborative: Finds similar books")
print("• User-based Collaborative: Finds similar users")
print("• Content-based: Uses book metadata similarity")

print("\n💡 Key Insights:")
print("• Most users give implicit feedback (rating=0)")
print("• Popular books dominate recommendations")
print("• Collaborative filtering works best for active users")
print("• Content-based helps with cold-start problems")

print("\n🚀 Production Considerations:")
print("• Filter active users and popular items")
print("• Use hybrid approaches for better results")
print("• Implement A/B testing for recommendation quality")
print("• Consider real-time updates and user feedback")