# ====================
# ADVANCED BOOK RECOMMENDATION MODELS
# ====================
# This script implements state-of-the-art recommendation algorithms including
# SVD, SVD++, NMF, and provides comprehensive model evaluation metrics.

import pandas as pd
import numpy as np
import pickle
from surprise import SVD, SVDpp, NMF, KNNBasic, KNNWithMeans
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class AdvancedBookRecommender:
    """
    Advanced recommendation system using matrix factorization techniques.
    Implements SVD, SVD++, NMF with hyperparameter tuning and evaluation.
    """

    def __init__(self, ratings_file='Ratings.csv'):
        """Initialize the recommender with ratings data."""
        print("Loading ratings data...")
        self.ratings_df = pd.read_csv(ratings_file, encoding='latin-1')
        print(f"Loaded {len(self.ratings_df):,} ratings")

        # Prepare data for Surprise library
        reader = Reader(rating_scale=(0, 10))
        self.data = Dataset.load_from_df(
            self.ratings_df[['User-ID', 'ISBN', 'Book-Rating']],
            reader
        )

        # Initialize models dictionary
        self.models = {}
        self.results = {}

    def prepare_data(self, test_size=0.2):
        """Split data into train and test sets."""
        print(f"\nSplitting data (test_size={test_size})...")
        self.trainset, self.testset = train_test_split(
            self.data,
            test_size=test_size,
            random_state=42
        )
        print(f"Training samples: {len(self.trainset.all_ratings()):,}")
        print(f"Testing samples: {len(self.testset):,}")

    def train_svd(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        """
        Train SVD (Singular Value Decomposition) model.

        SVD is the industry-standard matrix factorization technique used by Netflix.
        It factorizes the user-item matrix into latent factors.

        Parameters:
        - n_factors: Number of latent factors
        - n_epochs: Number of training iterations
        - lr_all: Learning rate
        - reg_all: Regularization parameter
        """
        print("\n" + "="*60)
        print("Training SVD Model")
        print("="*60)

        svd = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42,
            verbose=True
        )

        svd.fit(self.trainset)
        self.models['SVD'] = svd

        # Evaluate
        predictions = svd.test(self.testset)
        rmse = accuracy.rmse(predictions, verbose=True)
        mae = accuracy.mae(predictions, verbose=True)

        self.results['SVD'] = {
            'model': svd,
            'predictions': predictions,
            'RMSE': rmse,
            'MAE': mae
        }

        return svd

    def train_svdpp(self, n_factors=20, n_epochs=10):
        """
        Train SVD++ model.

        SVD++ extends SVD by taking into account implicit ratings.
        More accurate but slower than SVD.

        Note: Training is slower due to implicit feedback consideration.
        """
        print("\n" + "="*60)
        print("Training SVD++ Model (This may take longer...)")
        print("="*60)

        svdpp = SVDpp(
            n_factors=n_factors,
            n_epochs=n_epochs,
            random_state=42,
            verbose=True
        )

        svdpp.fit(self.trainset)
        self.models['SVDpp'] = svdpp

        # Evaluate
        predictions = svdpp.test(self.testset)
        rmse = accuracy.rmse(predictions, verbose=True)
        mae = accuracy.mae(predictions, verbose=True)

        self.results['SVDpp'] = {
            'model': svdpp,
            'predictions': predictions,
            'RMSE': rmse,
            'MAE': mae
        }

        return svdpp

    def train_nmf(self, n_factors=15, n_epochs=50):
        """
        Train NMF (Non-negative Matrix Factorization) model.

        NMF ensures all latent factors are non-negative, which can lead to
        more interpretable factors (e.g., genres, themes).
        """
        print("\n" + "="*60)
        print("Training NMF Model")
        print("="*60)

        nmf = NMF(
            n_factors=n_factors,
            n_epochs=n_epochs,
            random_state=42,
            verbose=True
        )

        nmf.fit(self.trainset)
        self.models['NMF'] = nmf

        # Evaluate
        predictions = nmf.test(self.testset)
        rmse = accuracy.rmse(predictions, verbose=True)
        mae = accuracy.mae(predictions, verbose=True)

        self.results['NMF'] = {
            'model': nmf,
            'predictions': predictions,
            'RMSE': rmse,
            'MAE': mae
        }

        return nmf

    def train_knn(self, k=40, sim_options=None):
        """
        Train KNN-based collaborative filtering.

        User-based or item-based collaborative filtering using k-nearest neighbors.
        """
        print("\n" + "="*60)
        print("Training KNN Model")
        print("="*60)

        if sim_options is None:
            sim_options = {
                'name': 'cosine',
                'user_based': False  # Item-based
            }

        knn = KNNWithMeans(k=k, sim_options=sim_options, verbose=True)
        knn.fit(self.trainset)
        self.models['KNN'] = knn

        # Evaluate
        predictions = knn.test(self.testset)
        rmse = accuracy.rmse(predictions, verbose=True)
        mae = accuracy.mae(predictions, verbose=True)

        self.results['KNN'] = {
            'model': knn,
            'predictions': predictions,
            'RMSE': rmse,
            'MAE': mae
        }

        return knn

    def hyperparameter_tuning(self, algorithm='SVD'):
        """
        Perform grid search for hyperparameter tuning.

        Finds optimal parameters for the specified algorithm using cross-validation.
        """
        print("\n" + "="*60)
        print(f"Hyperparameter Tuning for {algorithm}")
        print("="*60)

        if algorithm == 'SVD':
            param_grid = {
                'n_factors': [50, 100, 150],
                'n_epochs': [20, 30],
                'lr_all': [0.002, 0.005],
                'reg_all': [0.02, 0.1]
            }
            algo_class = SVD
        elif algorithm == 'NMF':
            param_grid = {
                'n_factors': [15, 20, 30],
                'n_epochs': [50, 100]
            }
            algo_class = NMF
        else:
            print(f"Hyperparameter tuning not configured for {algorithm}")
            return None

        gs = GridSearchCV(algo_class, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
        gs.fit(self.data)

        print(f"\nBest RMSE score: {gs.best_score['rmse']:.4f}")
        print("Best parameters:")
        for param, value in gs.best_params['rmse'].items():
            print(f"  {param}: {value}")

        return gs.best_params['rmse']

    def cross_validate_all(self):
        """Perform 5-fold cross-validation on all models."""
        print("\n" + "="*60)
        print("Cross-Validation (5-fold)")
        print("="*60)

        algorithms = {
            'SVD': SVD(n_factors=100, random_state=42),
            'SVDpp': SVDpp(n_factors=20, random_state=42),
            'NMF': NMF(n_factors=15, random_state=42),
            'KNN': KNNWithMeans(k=40)
        }

        cv_results = {}

        for name, algo in algorithms.items():
            print(f"\n{name}:")
            results = cross_validate(
                algo,
                self.data,
                measures=['RMSE', 'MAE'],
                cv=5,
                verbose=True
            )
            cv_results[name] = results

            print(f"  Mean RMSE: {results['test_rmse'].mean():.4f} (±{results['test_rmse'].std():.4f})")
            print(f"  Mean MAE:  {results['test_mae'].mean():.4f} (±{results['test_mae'].std():.4f})")

        return cv_results

    def get_top_n_recommendations(self, predictions, n=10):
        """
        Return the top-N recommendations for each user from a set of predictions.

        Args:
            predictions: List of prediction objects from surprise
            n: Number of recommendations per user

        Returns:
            Dictionary mapping user_id to list of (item_id, rating) tuples
        """
        top_n = defaultdict(list)

        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Sort predictions for each user and retrieve the k highest ones
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n

    def precision_recall_at_k(self, predictions, k=10, threshold=7):
        """
        Calculate Precision@K and Recall@K metrics.

        Args:
            predictions: List of predictions from model
            k: Number of recommendations to consider
            threshold: Minimum rating to consider as relevant

        Returns:
            Dictionary with precision and recall values
        """
        user_est_true = defaultdict(list)

        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = {}
        recalls = {}

        for uid, user_ratings in user_est_true.items():
            # Sort by estimated rating
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(
                ((true_r >= threshold) and (est >= threshold))
                for (est, true_r) in user_ratings[:k]
            )

            # Precision@K: Proportion of recommended items that are relevant
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            # Recall@K: Proportion of relevant items that are recommended
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        avg_precision = sum(precisions.values()) / len(precisions) if precisions else 0
        avg_recall = sum(recalls.values()) / len(recalls) if recalls else 0

        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        }

    def recommend_for_user(self, user_id, model_name='SVD', n=10):
        """
        Get top N book recommendations for a specific user.

        Args:
            user_id: User ID to generate recommendations for
            model_name: Name of the model to use ('SVD', 'SVDpp', 'NMF', 'KNN')
            n: Number of recommendations

        Returns:
            List of (ISBN, predicted_rating) tuples
        """
        if model_name not in self.models:
            print(f"Model {model_name} not trained yet!")
            return []

        model = self.models[model_name]

        # Get list of all books
        all_books = self.ratings_df['ISBN'].unique()

        # Get books already rated by user
        user_books = self.ratings_df[self.ratings_df['User-ID'] == user_id]['ISBN'].values

        # Books not yet rated
        books_to_predict = [book for book in all_books if book not in user_books]

        # Predict ratings
        predictions = []
        for book_isbn in books_to_predict:
            pred = model.predict(user_id, book_isbn)
            predictions.append((book_isbn, pred.est))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n]

    def save_models(self, filename='trained_models.pkl'):
        """Save all trained models to disk."""
        print(f"\nSaving models to {filename}...")
        with open(filename, 'wb') as f:
            pickle.dump(self.models, f)
        print("Models saved successfully!")

    def load_models(self, filename='trained_models.pkl'):
        """Load trained models from disk."""
        print(f"\nLoading models from {filename}...")
        with open(filename, 'rb') as f:
            self.models = pickle.load(f)
        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")

    def compare_models(self):
        """Generate a comprehensive comparison of all trained models."""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        if not self.results:
            print("No models trained yet!")
            return

        comparison = []
        for model_name, result in self.results.items():
            metrics = self.precision_recall_at_k(result['predictions'])
            comparison.append({
                'Model': model_name,
                'RMSE': result['RMSE'],
                'MAE': result['MAE'],
                'Precision@10': metrics['precision'],
                'Recall@10': metrics['recall'],
                'F1-Score': metrics['f1_score']
            })

        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.sort_values('RMSE')

        print("\n", df_comparison.to_string(index=False))

        # Highlight best model
        best_model = df_comparison.iloc[0]['Model']
        print(f"\n🏆 Best Model (lowest RMSE): {best_model}")

        return df_comparison


def main():
    """Main execution function."""
    print("="*60)
    print("ADVANCED BOOK RECOMMENDATION SYSTEM")
    print("="*60)

    # Initialize recommender
    recommender = AdvancedBookRecommender('Ratings.csv')

    # Prepare data
    recommender.prepare_data(test_size=0.2)

    # Train all models
    print("\n🚀 Training all models...")
    recommender.train_svd(n_factors=100, n_epochs=20)
    # recommender.train_svdpp(n_factors=20, n_epochs=10)  # Uncomment for SVD++ (slower)
    recommender.train_nmf(n_factors=15, n_epochs=50)
    recommender.train_knn(k=40)

    # Compare models
    comparison_df = recommender.compare_models()

    # Save comparison to CSV
    comparison_df.to_csv('model_comparison.csv', index=False)
    print("\n✅ Model comparison saved to 'model_comparison.csv'")

    # Save trained models
    recommender.save_models('trained_models.pkl')

    # Example: Get recommendations for a specific user
    print("\n" + "="*60)
    print("EXAMPLE RECOMMENDATIONS")
    print("="*60)

    # Get most active user
    most_active_user = recommender.ratings_df['User-ID'].value_counts().index[0]
    print(f"\nGenerating recommendations for User {most_active_user}:")

    recommendations = recommender.recommend_for_user(most_active_user, model_name='SVD', n=5)

    # Load books data to show titles
    try:
        books_df = pd.read_csv('Books.csv', encoding='latin-1')
        for i, (isbn, rating) in enumerate(recommendations, 1):
            book_info = books_df[books_df['ISBN'] == isbn]
            if not book_info.empty:
                title = book_info.iloc[0]['Book-Title']
                author = book_info.iloc[0]['Book-Author']
                print(f"{i}. {title} by {author}")
                print(f"   Predicted Rating: {rating:.2f}/10")
    except Exception as e:
        print("Could not load book details")
        for i, (isbn, rating) in enumerate(recommendations, 1):
            print(f"{i}. ISBN: {isbn} - Predicted Rating: {rating:.2f}")

    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
