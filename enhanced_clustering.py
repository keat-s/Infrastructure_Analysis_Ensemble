import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import os
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.inspection import permutation_importance
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import sys
import argparse
from sqlalchemy import create_engine, text
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_clustering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the original Autoencoder from neural_clustering.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from neural_clustering import Autoencoder
except ImportError:
    logger.warning("Could not import Autoencoder from neural_clustering.py, defining it here")
    
    class Autoencoder(nn.Module):
        """
        Autoencoder neural network for dimensionality reduction and feature learning.
        """
        def __init__(self, input_dim: int, encoding_dims: List[int], latent_dim: int):
            """
            Initialize the autoencoder architecture.
            
            Args:
                input_dim: Dimension of the input features
                encoding_dims: List of hidden layer dimensions for the encoder
                latent_dim: Dimension of the latent space representation
            """
            super(Autoencoder, self).__init__()
            
            # Build encoder layers
            encoder_layers = []
            prev_dim = input_dim
            
            for dim in encoding_dims:
                encoder_layers.append(nn.Linear(prev_dim, dim))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.BatchNorm1d(dim))
                prev_dim = dim
            
            # Final encoder layer to latent dimension
            encoder_layers.append(nn.Linear(prev_dim, latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)
            
            # Build decoder layers (mirror of encoder)
            decoder_layers = []
            prev_dim = latent_dim
            
            for dim in reversed(encoding_dims):
                decoder_layers.append(nn.Linear(prev_dim, dim))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.BatchNorm1d(dim))
                prev_dim = dim
            
            # Final decoder layer to original dimension
            decoder_layers.append(nn.Linear(prev_dim, input_dim))
            self.decoder = nn.Sequential(*decoder_layers)
        
        def forward(self, x):
            """Forward pass through the autoencoder."""
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
        
        def encode(self, x):
            """Encode the input to the latent space representation."""
            return self.encoder(x)


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for improved dimensionality reduction.
    
    VAEs learn a probabilistic mapping from input space to latent space,
    which can lead to better-behaved latent representations and more stable clustering.
    """
    def __init__(self, input_dim: int, encoding_dims: List[int], latent_dim: int):
        super(VariationalAutoencoder, self).__init__()
        
        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoding_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            encoder_layers.append(nn.BatchNorm1d(dim))
            prev_dim = dim
        
        self.encoder_layers = nn.Sequential(*encoder_layers)
        
        # Mean and variance layers
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_var = nn.Linear(prev_dim, latent_dim)
        
        # Build decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        
        for dim in reversed(encoding_dims):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            decoder_layers.append(nn.BatchNorm1d(dim))
            prev_dim = dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        """Encode input to get mu and log_var."""
        x = self.encoder_layers(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode from latent space to input space."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def get_latent_representation(self, x):
        """Get deterministic latent representation for clustering."""
        mu, _ = self.encode(x)
        return mu


class DeepClusteringEnsemble:
    """
    Enhanced deep clustering model that uses multiple techniques for improved performance.
    
    Key improvements:
    1. Ensemble of dimensionality reduction techniques
    2. Multiple clustering algorithms
    3. Feature selection and importance analysis
    4. Outlier detection and handling
    5. Advanced hyperparameter optimization
    6. Ensemble of clustering models
    7. Better validation metrics
    """
    def __init__(self, output_dir: Optional[str] = None, db_url: Optional[str] = None):
        """
        Initialize the enhanced deep clustering model.
        
        Args:
            output_dir: Directory for output files
            db_url: Database connection URL (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir:
            self.output_dir = os.path.join(output_dir, f"enhanced_clustering_{timestamp}")
        else:
            self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                         f"enhanced_clustering_{timestamp}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
        
        # Initialize database connection
        self.engine = None
        if db_url:
            try:
                logger.info("Initializing database connection...")
                self.engine = create_engine(db_url)
                # Test connection
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    logger.info("Database connection successful!")
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                self.engine = None
        else:
            # Try to use config if db_url not provided
            try:
                from config import Config
                db_url = Config.SQLALCHEMY_DATABASE_URI
                logger.info(f"Using database URL from config: {db_url.split('@')[1]}")
                self.engine = create_engine(db_url)
                # Test connection
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    logger.info("Database connection successful!")
            except Exception as e:
                logger.warning(f"Could not establish database connection from config: {e}")
                self.engine = None
                logger.warning("Warning: No database connection established. Will use CSV files if available.")
        
        # Save runtime parameters
        self.runtime_params = {
            'start_time': datetime.now(),
            'device': str(self.device),
            'output_dir': self.output_dir,
        }

    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load data either from a CSV file or from the database.
        
        Args:
            filepath: Path to CSV file (optional)
            
        Returns:
            DataFrame containing the server metrics data
        """
        if filepath and os.path.exists(filepath):
            logger.info(f"Loading data from file: {filepath}")
            return pd.read_csv(filepath)
        
        if self.engine:
            try:
                logger.info("Attempting to load data from database...")
                # First try to query the server_analysis_summary table if it exists
                try:
                    query = text("SELECT * FROM server_analysis_summary")
                    df = pd.read_sql_query(query, self.engine)
                    logger.info(f"Successfully loaded {len(df)} records from server_analysis_summary table")
                    return df
                except Exception as e:
                    logger.warning(f"Could not query server_analysis_summary: {e}")
                    
                    # If that fails, try a more generic query on servers and projects tables
                    # [Same query logic as in original neural_clustering.py]
                    # ...
            except Exception as db_error:
                logger.error(f"Database connection error: {db_error}")
        
        # Default to using the CSV in the clustering directory if no other source
        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server_resource_metrics.csv')
        if os.path.exists(default_path):
            logger.info(f"Falling back to CSV file: {default_path}")
            return pd.read_csv(default_path)
        
        raise FileNotFoundError("No data source available. Please provide a filepath or database connection.")

    def detect_and_handle_outliers(self, df: pd.DataFrame, 
                                 contamination: float = 0.05) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Detect and handle outliers in the dataset.
        
        Args:
            df: DataFrame with numeric features
            contamination: Expected proportion of outliers
            
        Returns:
            Tuple of (processed_df, outlier_mask)
        """
        logger.info("Detecting outliers...")
        
        # Only use numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        # Fill missing values with median (more robust than mean for outlier detection)
        filled_df = numeric_df.fillna(numeric_df.median())
        
        # Use Isolation Forest for outlier detection (robust to high-dimensional data)
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(filled_df)
        
        # Convert to boolean mask (True for inliers, False for outliers)
        inlier_mask = outlier_labels == 1
        outlier_mask = ~inlier_mask
        
        # Count outliers
        n_outliers = np.sum(outlier_mask)
        logger.info(f"Detected {n_outliers} outliers ({n_outliers/len(df):.1%} of data)")
        
        if n_outliers > 0:
            # Save outlier information
            outlier_df = df.copy()
            outlier_df['is_outlier'] = outlier_mask
            outlier_df[outlier_mask].to_csv(os.path.join(self.output_dir, 'detected_outliers.csv'), index=False)
            
            # Plot outlier scores
            plt.figure(figsize=(10, 6))
            outlier_scores = iso_forest.decision_function(filled_df)
            plt.hist(outlier_scores, bins=50)
            
            # Calculate threshold from scores instead of using threshold_ attribute
            # The threshold is the score at the contamination percentile
            threshold = np.percentile(outlier_scores, contamination * 100)
            
            plt.axvline(x=threshold, color='r', linestyle='--', 
                       label=f'Threshold ({threshold:.3f})')
            plt.title('Outlier Score Distribution')
            plt.xlabel('Outlier Score')
            plt.ylabel('Count')
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, 'outlier_scores.png'))
            plt.close()
            
            # Visualization of outliers in 2D
            if filled_df.shape[1] > 2:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(filled_df)
                
                plt.figure(figsize=(10, 8))
                plt.scatter(pca_result[inlier_mask, 0], pca_result[inlier_mask, 1], 
                           c='blue', label='Inliers', alpha=0.5)
                plt.scatter(pca_result[outlier_mask, 0], pca_result[outlier_mask, 1], 
                           c='red', label='Outliers', alpha=0.7)
                plt.title('PCA Visualization of Outliers')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.legend()
                plt.savefig(os.path.join(self.output_dir, 'outliers_pca.png'))
                plt.close()
        
        return df, outlier_mask

    def select_features(self, X: np.ndarray, feature_names: List[str], 
                      method: str = 'variance', threshold: float = 0.01, 
                      k: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Select the most important features for clustering.
        
        Args:
            X: Input data matrix
            feature_names: List of feature names
            method: Feature selection method ('variance', 'kbest')
            threshold: Threshold for variance-based selection
            k: Number of features to select for k-best methods
            
        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        logger.info(f"Performing feature selection using {method} method...")
        
        # Determine k if not provided
        if k is None:
            k = max(int(X.shape[1] * 0.7), 5)  # Select 70% of features or at least 5
        
        if method == 'variance':
            # Remove low-variance features
            selector = VarianceThreshold(threshold=threshold)
            X_selected = selector.fit_transform(X)
            mask = selector.get_support()
            selected_names = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
            
            logger.info(f"Selected {X_selected.shape[1]} features using variance threshold {threshold}")
            
        elif method == 'kbest':
            # First cluster the data to get pseudo-labels
            kmeans = KMeans(n_clusters=min(5, X.shape[0] - 1), random_state=42, n_init=10)
            pseudo_labels = kmeans.fit_predict(X)
            
            # Select k best features based on ANOVA F-statistic
            selector = SelectKBest(f_classif, k=k)
            X_selected = selector.fit_transform(X, pseudo_labels)
            mask = selector.get_support()
            selected_names = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
            
            # Get feature scores
            scores = selector.scores_
            feature_scores = list(zip(feature_names, scores))
            sorted_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            selected_scores = [score for name, score in sorted_features if name in selected_names]
            selected_names_sorted = [name for name, _ in sorted_features if name in selected_names]
            
            plt.barh(range(len(selected_scores)), selected_scores, align='center')
            plt.yticks(range(len(selected_scores)), selected_names_sorted)
            plt.xlabel('F-Score')
            plt.title(f'Top {k} Features by F-Score')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
            plt.close()
            
            logger.info(f"Selected top {k} features using k-best method")
            
        elif method == 'mutual_info':
            # First cluster the data to get pseudo-labels
            kmeans = KMeans(n_clusters=min(5, X.shape[0] - 1), random_state=42, n_init=10)
            pseudo_labels = kmeans.fit_predict(X)
            
            # Select k best features based on mutual information
            selector = SelectKBest(mutual_info_classif, k=k)
            X_selected = selector.fit_transform(X, pseudo_labels)
            mask = selector.get_support()
            selected_names = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
            
            logger.info(f"Selected top {k} features using mutual information")
            
        else:
            logger.warning(f"Unknown feature selection method: {method}. Using all features.")
            X_selected = X
            selected_names = feature_names
        
        # Log the selected features
        logger.info(f"Selected features: {', '.join(selected_names)}")
        
        return X_selected, selected_names

    def preprocess_data(self, df: pd.DataFrame, 
                       scaler_type: str = 'robust',
                       handle_outliers: bool = True,
                       feature_selection: str = 'kbest') -> Dict:
        """
        Enhanced preprocessing pipeline for the data.
        
        Args:
            df: DataFrame containing the server metrics
            scaler_type: Type of scaler to use ('standard', 'minmax', 'robust')
            handle_outliers: Whether to detect and handle outliers
            feature_selection: Feature selection method ('variance', 'kbest', 'mutual_info', None)
            
        Returns:
            Dictionary with preprocessing results
        """
        logger.info("Preprocessing data...")
        
        # 1. Extract numeric features and handle missing values
        numeric_df = df.select_dtypes(include=['number'])
        numeric_df = numeric_df.fillna(numeric_df.median())
        
        # Store feature names
        feature_names = numeric_df.columns.tolist()
        logger.info(f"Working with {len(feature_names)} numeric features")
        
        # 2. Outlier detection and handling
        outlier_mask = None
        if handle_outliers:
            _, outlier_mask = self.detect_and_handle_outliers(df, contamination=0.05)
            
            # We'll keep outliers in the dataset but mark them
            df['is_outlier'] = outlier_mask
            
            # For certain analyses, we might want to exclude outliers
            # numeric_df = numeric_df[~outlier_mask]
        
        # 3. Scale the data
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaler type: {scaler_type}. Using RobustScaler.")
            scaler = RobustScaler()
        
        scaled_data = scaler.fit_transform(numeric_df)
        
        # 4. Feature selection
        selected_data = scaled_data
        selected_feature_names = feature_names
        
        if feature_selection:
            selected_data, selected_feature_names = self.select_features(
                scaled_data, feature_names, method=feature_selection)
        
        # 5. Data quality summary
        # Check for highly correlated features
        corr_matrix = numeric_df[selected_feature_names].corr()
        high_corr_pairs = []
        
        for i in range(len(selected_feature_names)):
            for j in range(i+1, len(selected_feature_names)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((
                        selected_feature_names[i], 
                        selected_feature_names[j], 
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
            for f1, f2, corr in high_corr_pairs[:5]:  # Show only top 5
                logger.info(f"  {f1} and {f2}: {corr:.3f}")
            
            # Visualize correlation matrix
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                       annot=False, square=True, linewidths=.5)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_correlation.png'))
            plt.close()
        
        # Return all preprocessing results
        return {
            'scaled_data': scaled_data,
            'selected_data': selected_data,
            'feature_names': feature_names,
            'selected_feature_names': selected_feature_names,
            'scaler': scaler,
            'outlier_mask': outlier_mask,
            'high_correlation_pairs': high_corr_pairs
        }

    def train_dimensionality_reduction_models(self, 
                                            X: np.ndarray, 
                                            methods: List[str] = ['pca', 'vae', 'kpca'],
                                            latent_dim: int = 8) -> Dict:
        """
        Train multiple dimensionality reduction models and combine their results.
        
        Args:
            X: Input data (scaled)
            methods: List of dimensionality reduction methods to use
            latent_dim: Dimension of the latent space
            
        Returns:
            Dictionary with trained models and encoded features
        """
        logger.info(f"Training dimensionality reduction models: {', '.join(methods)}")
        
        results = {}
        all_encodings = []
        
        # 1. PCA - Linear dimensionality reduction
        if 'pca' in methods:
            logger.info(f"Training PCA model with {latent_dim} components...")
            pca = PCA(n_components=latent_dim, random_state=42)
            pca_result = pca.fit_transform(X)
            
            results['pca'] = {
                'model': pca,
                'encoded': pca_result,
                'explained_variance': pca.explained_variance_ratio_.sum()
            }
            
            all_encodings.append(pca_result)
            
            # Plot explained variance
            plt.figure(figsize=(10, 6))
            plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
            plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% Explained Variance')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('PCA Explained Variance')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, 'pca_explained_variance.png'))
            plt.close()
            
            logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        # 2. Kernel PCA - Non-linear dimensionality reduction
        if 'kpca' in methods:
            logger.info(f"Training Kernel PCA model with {latent_dim} components...")
            kpca = KernelPCA(n_components=latent_dim, kernel='rbf', random_state=42)
            kpca_result = kpca.fit_transform(X)
            
            results['kpca'] = {
                'model': kpca,
                'encoded': kpca_result
            }
            
            all_encodings.append(kpca_result)
        
        # 3. TruncatedSVD - Works well with sparse data
        if 'svd' in methods:
            logger.info(f"Training TruncatedSVD model with {latent_dim} components...")
            svd = TruncatedSVD(n_components=latent_dim, random_state=42)
            svd_result = svd.fit_transform(X)
            
            results['svd'] = {
                'model': svd,
                'encoded': svd_result,
                'explained_variance': svd.explained_variance_ratio_.sum()
            }
            
            all_encodings.append(svd_result)
            
            logger.info(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.2%}")
        
        # 4. Autoencoder - Neural network-based dimensionality reduction
        if 'ae' in methods:
            logger.info("Training Autoencoder model...")
            X_tensor = torch.FloatTensor(X)
            
            # Define encoding dimensions
            encoding_dims = [max(X.shape[1] // 2, latent_dim * 2), latent_dim * 2]
            
            # Initialize model
            model = Autoencoder(X.shape[1], encoding_dims, latent_dim).to(self.device)
            
            # Train the model
            ae_model, ae_encoded = self._train_autoencoder(X, model, batch_size=min(16, len(X)))
            
            results['ae'] = {
                'model': ae_model,
                'encoded': ae_encoded
            }
            
            all_encodings.append(ae_encoded)
        
        # 5. Variational Autoencoder - Probabilistic dimensionality reduction
        if 'vae' in methods:
            logger.info("Training Variational Autoencoder model...")
            
            # Define encoding dimensions
            encoding_dims = [max(X.shape[1] // 2, latent_dim * 2), latent_dim * 2]
            
            # Initialize model
            vae_model = VariationalAutoencoder(X.shape[1], encoding_dims, latent_dim).to(self.device)
            
            # Train the model
            vae_model, vae_encoded = self._train_vae(X, vae_model, batch_size=min(16, len(X)))
            
            results['vae'] = {
                'model': vae_model,
                'encoded': vae_encoded
            }
            
            all_encodings.append(vae_encoded)
        
        # 6. UMAP - Manifold learning for visualization and dimensionality reduction
        if 'umap' in methods:
            try:
                import umap
                logger.info("Training UMAP model...")
                
                umap_model = umap.UMAP(n_components=latent_dim, random_state=42)
                umap_result = umap_model.fit_transform(X)
                
                results['umap'] = {
                    'model': umap_model,
                    'encoded': umap_result
                }
                
                all_encodings.append(umap_result)
            except ImportError:
                logger.warning("UMAP not installed. Skipping UMAP dimensionality reduction.")
        
        # 7. Combine encodings if we have multiple methods
        if len(all_encodings) > 1:
            # Normalize each encoding before concatenation
            normalized_encodings = []
            for encoding in all_encodings:
                scaler = StandardScaler()
                normalized_encodings.append(scaler.fit_transform(encoding))
            
            # Concatenate all encodings
            combined_encoding = np.hstack(normalized_encodings)
            
            # Use PCA to reduce dimensionality of combined encoding
            final_dim = min(latent_dim * 2, combined_encoding.shape[1])
            combined_pca = PCA(n_components=final_dim, random_state=42)
            combined_result = combined_pca.fit_transform(combined_encoding)
            
            results['combined'] = {
                'model': combined_pca,
                'encoded': combined_result,
                'source_models': methods
            }
            
            logger.info(f"Created combined encoding with dimensionality {final_dim}")
        
        return results

    def _train_autoencoder(self, 
                         X: np.ndarray, 
                         model: nn.Module,
                         batch_size: int = 16, 
                         epochs: int = 200, 
                         learning_rate: float = 0.001) -> Tuple[nn.Module, np.ndarray]:
        """
        Train an autoencoder model.
        
        Args:
            X: Input data (scaled)
            model: Autoencoder model
            batch_size: Batch size
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            
        Returns:
            Tuple of (trained_model, encoded_features)
        """
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor, X_tensor)  # Input equals target for autoencoder
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                
                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
        
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Autoencoder Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.savefig(os.path.join(self.output_dir, 'autoencoder_training_loss.png'))
        plt.close()
        
        # Generate encoded features
        model.eval()
        with torch.no_grad():
            encoded_features = model.encode(X_tensor.to(self.device)).cpu().numpy()
        
        return model, encoded_features

    def _train_vae(self, 
                  X: np.ndarray, 
                  model: VariationalAutoencoder,
                  batch_size: int = 16, 
                  epochs: int = 200, 
                  learning_rate: float = 0.001,
                  kl_weight: float = 0.001) -> Tuple[nn.Module, np.ndarray]:
        """
        Train a variational autoencoder model.
        
        Args:
            X: Input data (scaled)
            model: VAE model
            batch_size: Batch size
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            kl_weight: Weight for KL divergence loss term
            
        Returns:
            Tuple of (trained_model, encoded_features)
        """
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        losses = []
        rec_losses = []
        kl_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_rec_loss = 0
            epoch_kl_loss = 0
            
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                
                # Forward pass
                recon_x, mu, log_var = model(batch_x)
                
                # Calculate loss
                recon_loss = F.mse_loss(recon_x, batch_x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + kl_weight * kl_loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track losses
                epoch_loss += loss.item()
                epoch_rec_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
            
            # Average losses
            avg_loss = epoch_loss / len(dataloader)
            avg_rec_loss = epoch_rec_loss / len(dataloader)
            avg_kl_loss = epoch_kl_loss / len(dataloader)
            
            losses.append(avg_loss)
            rec_losses.append(avg_rec_loss)
            kl_losses.append(avg_kl_loss)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, '
                          f'Recon Loss: {avg_rec_loss:.6f}, KL Loss: {avg_kl_loss:.6f}')
        
        # Plot training losses
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Total Loss')
        plt.plot(rec_losses, label='Reconstruction Loss')
        plt.title('VAE Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(kl_losses)
        plt.title('KL Divergence Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'vae_training_loss.png'))
        plt.close()
        
        # Generate encoded features (use mean vectors from encoder)
        model.eval()
        with torch.no_grad():
            encoded_features = model.get_latent_representation(X_tensor.to(self.device)).cpu().numpy()
        
        return model, encoded_features

    def perform_ensemble_clustering(self, 
                                  encoded_features_dict: Dict, 
                                  max_clusters: int = 12,
                                  min_clusters: int = 2) -> Dict:
        """
        Perform ensemble clustering using multiple clustering algorithms and 
        encoded features from different dimensionality reduction methods.
        
        Args:
            encoded_features_dict: Dictionary with encoded features from different models
            max_clusters: Maximum number of clusters to try
            min_clusters: Minimum number of clusters to try
            
        Returns:
            Dictionary with clustering results
        """
        logger.info("Performing ensemble clustering...")
        
        # Define clustering algorithms to try
        clustering_algorithms = {
            'kmeans': KMeans,
            'agglomerative': AgglomerativeClustering,
            'spectral': SpectralClustering,
            'birch': Birch
        }
        
        # Store all clustering results
        all_results = {}
        best_silhouette = -1
        best_result = None
        best_algo = None
        best_encoding = None
        best_n_clusters = None
        
        # For each encoding method
        for encoding_name, encoding_info in encoded_features_dict.items():
            encoded_features = encoding_info['encoded']
            
            logger.info(f"Evaluating clustering on {encoding_name} encoding...")
            
            # Determine optimal number of clusters for this encoding
            optimal_n, silhouette_scores = self._determine_optimal_clusters(
                encoded_features, max_clusters=max_clusters, min_clusters=min_clusters)
            
            # Try different clustering algorithms with optimal cluster number
            for algo_name, algo_class in clustering_algorithms.items():
                try:
                    logger.info(f"Trying {algo_name} clustering with {optimal_n} clusters...")
                    
                    # Initialize and fit clustering algorithm
                    if algo_name == 'kmeans':
                        clusterer = algo_class(n_clusters=optimal_n, random_state=42, n_init=10)
                    elif algo_name == 'agglomerative':
                        clusterer = algo_class(n_clusters=optimal_n, linkage='ward')
                    elif algo_name == 'spectral':
                        # Skip spectral for larger datasets as it's computationally expensive
                        if len(encoded_features) > 200:
                            logger.warning(f"Skipping spectral clustering for large dataset")
                            continue
                        clusterer = algo_class(n_clusters=optimal_n, random_state=42, 
                                             affinity='nearest_neighbors')
                    elif algo_name == 'birch':
                        clusterer = algo_class(n_clusters=optimal_n)
                    
                    # Fit and predict
                    cluster_labels = clusterer.fit_predict(encoded_features)
                    
                    # Calculate validation metrics
                    sil_score = silhouette_score(encoded_features, cluster_labels)
                    ch_score = calinski_harabasz_score(encoded_features, cluster_labels)
                    db_score = davies_bouldin_score(encoded_features, cluster_labels)
                    
                    # Store results
                    result = {
                        'cluster_labels': cluster_labels,
                        'n_clusters': optimal_n,
                        'silhouette_score': sil_score,
                        'calinski_harabasz_score': ch_score,
                        'davies_bouldin_score': db_score,
                        'encoding': encoding_name,
                        'algorithm': algo_name,
                        'clusterer': clusterer
                    }
                    
                    # Add to results dictionary
                    key = f"{encoding_name}_{algo_name}"
                    all_results[key] = result
                    
                    logger.info(f"  {key}: Silhouette={sil_score:.4f}, CH={ch_score:.1f}, DB={db_score:.4f}")
                    
                    # Update best result if current is better
                    if sil_score > best_silhouette:
                        best_silhouette = sil_score
                        best_result = result
                        best_algo = algo_name
                        best_encoding = encoding_name
                        best_n_clusters = optimal_n
                        
                except Exception as e:
                    logger.error(f"Error in {algo_name} clustering: {e}")
        
        # Generate ensemble clustering (consensus clustering)
        if len(all_results) > 1:
            logger.info("Generating ensemble clustering from all methods...")
            
            # Collect all cluster labels
            all_labels = np.column_stack([
                result['cluster_labels'] for result in all_results.values()
            ])
            
            # Convert to similarity matrix
            n_samples = all_labels.shape[0]
            similarity_matrix = np.zeros((n_samples, n_samples))
            
            for i in range(n_samples):
                for j in range(i, n_samples):
                    # Count how many times samples i and j are clustered together
                    same_cluster = np.sum(all_labels[i] == all_labels[j])
                    similarity = same_cluster / all_labels.shape[1]
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
            
            # Use hierarchical clustering on the similarity matrix
            try:
                # Convert similarity to distance
                distance_matrix = 1 - similarity_matrix
                
                # Try different parameter combinations for AgglomerativeClustering
                try:
                    # First try with affinity='precomputed' and linkage='average'
                    ensemble_clusterer = AgglomerativeClustering(
                        n_clusters=best_n_clusters,
                        affinity='precomputed',
                        linkage='average'
                    )
                    ensemble_labels = ensemble_clusterer.fit_predict(distance_matrix)
                except TypeError:
                    try:
                        # If that fails, try with just metric='precomputed'
                        ensemble_clusterer = AgglomerativeClustering(
                            n_clusters=best_n_clusters,
                            metric='precomputed',
                            linkage='average'
                        )
                        ensemble_labels = ensemble_clusterer.fit_predict(distance_matrix)
                    except TypeError:
                        # If that also fails, use KMeans as a fallback
                        logger.warning("AgglomerativeClustering with precomputed distances not supported. Using KMeans instead.")
                        # Apply PCA to reduce dimensionality of similarity matrix
                        pca = PCA(n_components=min(n_samples-1, 10))
                        similarity_features = pca.fit_transform(similarity_matrix)
                        ensemble_clusterer = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
                        ensemble_labels = ensemble_clusterer.fit_predict(similarity_features)
                
                # Calculate validation metrics for ensemble result
                # We'll use the best encoding for validation
                best_encoding_features = encoded_features_dict[best_encoding]['encoded']
                
                ensemble_sil = silhouette_score(best_encoding_features, ensemble_labels)
                ensemble_ch = calinski_harabasz_score(best_encoding_features, ensemble_labels)
                ensemble_db = davies_bouldin_score(best_encoding_features, ensemble_labels)
                
                ensemble_result = {
                    'cluster_labels': ensemble_labels,
                    'n_clusters': best_n_clusters,
                    'silhouette_score': ensemble_sil,
                    'calinski_harabasz_score': ensemble_ch,
                    'davies_bouldin_score': ensemble_db,
                    'encoding': 'ensemble',
                    'algorithm': 'consensus',
                    'similarity_matrix': similarity_matrix
                }
                
                all_results['ensemble'] = ensemble_result
                
                logger.info(f"Ensemble clustering: Silhouette={ensemble_sil:.4f}, "
                          f"CH={ensemble_ch:.1f}, DB={ensemble_db:.4f}")
                
                # Update best result if ensemble is better
                if ensemble_sil > best_silhouette:
                    best_silhouette = ensemble_sil
                    best_result = ensemble_result
                    best_algo = 'consensus'
                    best_encoding = 'ensemble'
            
            except Exception as e:
                logger.error(f"Error in ensemble clustering: {e}")
                logger.warning("Skipping ensemble clustering due to error.")
        
        logger.info(f"Best clustering: {best_encoding}+{best_algo} with "
                  f"{best_n_clusters} clusters, Silhouette={best_silhouette:.4f}")
        
        # Compare all methods visually
        self._visualize_clustering_comparison(encoded_features_dict, all_results)
        
        return {
            'all_results': all_results,
            'best_result': best_result,
            'best_algorithm': best_algo,
            'best_encoding': best_encoding
        }

    def _determine_optimal_clusters(self, 
                                 X: np.ndarray, 
                                 max_clusters: int = 12,
                                 min_clusters: int = 2) -> Tuple[int, List[float]]:
        """
        Determine the optimal number of clusters using multiple metrics.
        
        Args:
            X: Feature matrix (encoded features)
            max_clusters: Maximum number of clusters to try
            min_clusters: Minimum number of clusters to try
            
        Returns:
            Tuple of (optimal_clusters, silhouette_scores)
        """
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        
        # Limit max_clusters to one less than the number of samples
        max_clusters = min(max_clusters, X.shape[0] - 1)
        
        # Try different numbers of clusters
        cluster_range = range(min_clusters, max_clusters + 1)
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate silhouette score
            sil_score = silhouette_score(X, cluster_labels)
            silhouette_scores.append(sil_score)
            
            # Calculate Calinski-Harabasz score
            ch_score = calinski_harabasz_score(X, cluster_labels)
            calinski_scores.append(ch_score)
            
            # Calculate Davies-Bouldin score (lower is better)
            db_score = davies_bouldin_score(X, cluster_labels)
            davies_bouldin_scores.append(db_score)
            
            logger.info(f"Clusters: {n_clusters}, Silhouette: {sil_score:.4f}, "
                      f"CH: {ch_score:.1f}, DB: {db_score:.4f}")
        
        # Plot the scores
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(cluster_range, silhouette_scores, marker='o')
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(cluster_range, calinski_scores, marker='o')
        plt.title('Calinski-Harabasz Score vs. Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Calinski-Harabasz Score')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(cluster_range, davies_bouldin_scores, marker='o')
        plt.title('Davies-Bouldin Score vs. Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Davies-Bouldin Score (lower is better)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_evaluation_metrics.png'))
        plt.close()
        
        # Determine optimal clusters
        # 1. Find local maxima in silhouette scores
        # A local maximum needs to be greater than its neighbors
        silhouette_local_maxima = []
        for i in range(1, len(silhouette_scores) - 1):
            if (silhouette_scores[i] > silhouette_scores[i-1] and 
                silhouette_scores[i] > silhouette_scores[i+1]):
                silhouette_local_maxima.append((i + min_clusters, silhouette_scores[i]))
        
        # 2. Find local minima in Davies-Bouldin index (lower is better)
        db_local_minima = []
        for i in range(1, len(davies_bouldin_scores) - 1):
            if (davies_bouldin_scores[i] < davies_bouldin_scores[i-1] and 
                davies_bouldin_scores[i] < davies_bouldin_scores[i+1]):
                db_local_minima.append((i + min_clusters, davies_bouldin_scores[i]))
        
        # Sort maxima/minima by score
        silhouette_local_maxima.sort(key=lambda x: x[1], reverse=True)
        db_local_minima.sort(key=lambda x: x[1])
        
        # Combine evidence from different metrics
        optimal_clusters = min_clusters
        
        # First, check if we have local maxima/minima to consider
        if silhouette_local_maxima:
            best_silhouette_clusters = silhouette_local_maxima[0][0]
            optimal_clusters = best_silhouette_clusters
            
            # See if DB index agrees with a similar number
            if db_local_minima:
                best_db_clusters = db_local_minima[0][0]
                
                # If the two metrics suggest similar numbers, choose the silhouette one
                # Otherwise, prefer more clusters if scores are close enough
                if abs(best_silhouette_clusters - best_db_clusters) <= 1:
                    optimal_clusters = best_silhouette_clusters
                else:
                    # Check if any DB-suggested cluster number has a good silhouette score
                    for n_clusters, _ in db_local_minima:
                        idx = n_clusters - min_clusters
                        if idx < len(silhouette_scores):
                            sil_score = silhouette_scores[idx]
                            if sil_score >= max(silhouette_scores) * 0.9:  # Within 90% of max
                                optimal_clusters = n_clusters
                                break
        else:
            # If no local maxima, use the global maximum
            optimal_clusters = min_clusters + np.argmax(silhouette_scores)
        
        logger.info(f"Selected optimal number of clusters: {optimal_clusters}")
        return optimal_clusters, silhouette_scores

    def _visualize_clustering_comparison(self, 
                                       encoded_features_dict: Dict, 
                                       clustering_results: Dict):
        """
        Visualize and compare different clustering results.
        
        Args:
            encoded_features_dict: Dictionary with encoded features
            clustering_results: Dictionary with clustering results
        """
        # Use PCA or t-SNE to visualize results in 2D
        plt.figure(figsize=(15, 10))
        n_methods = min(6, len(clustering_results))  # Show max 6 methods
        
        # Choose top methods by silhouette score
        top_methods = sorted(clustering_results.items(), 
                            key=lambda x: x[1]['silhouette_score'], 
                            reverse=True)[:n_methods]
        
        for i, (name, result) in enumerate(top_methods):
            # Get the corresponding encoded features
            encoding_name = result['encoding']
            if encoding_name == 'ensemble':
                # For ensemble, use the best encoding for visualization
                best_encoding = None
                best_sil = -1
                for enc_name, enc_info in encoded_features_dict.items():
                    if enc_name != 'combined':  # Skip combined encoding
                        labels = result['cluster_labels']
                        features = enc_info['encoded']
                        sil = silhouette_score(features, labels)
                        if sil > best_sil:
                            best_sil = sil
                            best_encoding = enc_name
                
                encoded_features = encoded_features_dict[best_encoding]['encoded']
            else:
                encoded_features = encoded_features_dict[encoding_name]['encoded']
            
            # Use t-SNE for visualization if dimension > 2
            if encoded_features.shape[1] > 2:
                tsne = TSNE(n_components=2, random_state=42)
                viz_data = tsne.fit_transform(encoded_features)
            else:
                viz_data = encoded_features
            
            # Plot
            plt.subplot(2, 3, i+1)
            plt.scatter(viz_data[:, 0], viz_data[:, 1], 
                       c=result['cluster_labels'], cmap='viridis', 
                       s=50, alpha=0.8)
            
            # Add title with metrics
            title = f"{name}\nSilhouette: {result['silhouette_score']:.3f}"
            if 'calinski_harabasz_score' in result:
                title += f"\nCH: {result['calinski_harabasz_score']:.1f}"
            if 'davies_bouldin_score' in result:
                title += f"\nDB: {result['davies_bouldin_score']:.3f}"
            
            plt.title(title)
            plt.colorbar(label=f"Cluster ({result['n_clusters']})")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'clustering_comparison.png'))
        plt.close()
        
        # Create cluster size comparison
        plt.figure(figsize=(15, 5))
        
        for i, (name, result) in enumerate(top_methods[:3]):  # Show top 3
            plt.subplot(1, 3, i+1)
            
            # Count clusters
            unique_labels, counts = np.unique(result['cluster_labels'], return_counts=True)
            
            # Sort by size
            sorted_idx = np.argsort(counts)[::-1]
            counts = counts[sorted_idx]
            unique_labels = unique_labels[sorted_idx]
            
            # Plot cluster sizes
            plt.bar(range(len(counts)), counts)
            plt.xticks(range(len(counts)), unique_labels)
            plt.title(f"{name}\nCluster Size Distribution")
            plt.xlabel("Cluster ID")
            plt.ylabel("Number of Samples")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_size_comparison.png'))
        plt.close()

    def validate_clusters(self, 
                        X: np.ndarray, 
                        best_clustering: Dict,
                        n_splits: int = 5) -> Dict:
        """
        Perform comprehensive validation of the clustering results.
        
        Args:
            X: Original feature matrix (scaled)
            best_clustering: Dictionary with the best clustering result
            n_splits: Number of cross-validation splits
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info("Performing comprehensive cluster validation...")
        
        n_clusters = best_clustering['n_clusters']
        cluster_labels = best_clustering['cluster_labels']
        
        # 1. Stability analysis via cross-validation
        logger.info(f"Performing {n_splits}-fold cross-validation for cluster stability...")
        
        # Initialize cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Store validation metrics
        stability_metrics = []
        robustness_metrics = []
        
        # Get the clustering algorithm used
        algo_name = best_clustering['algorithm']
        
        # For consensus clustering, we'll use the best individual algorithm
        if algo_name == 'consensus':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:
            # Create a fresh instance of the same algorithm
            if algo_name == 'kmeans':
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            elif algo_name == 'agglomerative':
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            elif algo_name == 'spectral':
                clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42, 
                                             affinity='nearest_neighbors')
            elif algo_name == 'birch':
                clusterer = Birch(n_clusters=n_clusters)
            else:
                # Default to KMeans
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            # Use training set to fit clustering
            if algo_name in ('kmeans', 'birch'):
                # These algorithms support partial_fit or can directly predict on new data
                train_features = X[train_idx]
                test_features = X[test_idx]
                
                clusterer.fit(train_features)
                test_labels = clusterer.predict(test_features)
            else:
                # For algorithms that don't support predict(), fit on test data
                test_features = X[test_idx]
                if algo_name == 'agglomerative':
                    test_clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                    test_labels = test_clusterer.fit_predict(test_features)
                elif algo_name == 'spectral':
                    test_clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42, 
                                                      affinity='nearest_neighbors')
                    test_labels = test_clusterer.fit_predict(test_features)
                else:
                    # Fall back to KMeans for unknown algorithms
                    test_clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    test_labels = test_clusterer.fit_predict(test_features)
            
            # Get original labels for test data
            original_test_labels = cluster_labels[test_idx]
            
            # Compute stability metrics
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            
            ari = adjusted_rand_score(original_test_labels, test_labels)
            nmi = normalized_mutual_info_score(original_test_labels, test_labels)
            
            # Calculate silhouette score on test set
            if len(np.unique(test_labels)) > 1:
                sil = silhouette_score(test_features, test_labels)
            else:
                sil = 0
            
            stability_metrics.append((ari, nmi, sil))
            
            logger.info(f"Fold {fold+1}: ARI = {ari:.4f}, NMI = {nmi:.4f}, Silhouette = {sil:.4f}")
            
            # Robustness to noise
            # Add small Gaussian noise to the data
            noise_level = 0.05
            noisy_features = test_features + np.random.normal(0, noise_level, test_features.shape)
            
            # Cluster the noisy data
            if algo_name in ('kmeans', 'birch'):
                noisy_labels = clusterer.predict(noisy_features)
            else:
                if algo_name == 'agglomerative':
                    noisy_clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                    noisy_labels = noisy_clusterer.fit_predict(noisy_features)
                elif algo_name == 'spectral':
                    noisy_clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42, 
                                                       affinity='nearest_neighbors')
                    noisy_labels = noisy_clusterer.fit_predict(noisy_features)
                else:
                    noisy_clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    noisy_labels = noisy_clusterer.fit_predict(noisy_features)
            
            # Compare with clean clustering
            noise_ari = adjusted_rand_score(test_labels, noisy_labels)
            noise_nmi = normalized_mutual_info_score(test_labels, noisy_labels)
            
            robustness_metrics.append((noise_ari, noise_nmi))
            
            logger.info(f"  Noise robustness: ARI = {noise_ari:.4f}, NMI = {noise_nmi:.4f}")
        
        # Calculate average metrics
        avg_ari = np.mean([m[0] for m in stability_metrics])
        avg_nmi = np.mean([m[1] for m in stability_metrics])
        avg_sil = np.mean([m[2] for m in stability_metrics])
        
        avg_noise_ari = np.mean([m[0] for m in robustness_metrics])
        avg_noise_nmi = np.mean([m[1] for m in robustness_metrics])
        
        logger.info(f"Average validation metrics:")
        logger.info(f"  Stability - ARI: {avg_ari:.4f}, NMI: {avg_nmi:.4f}, Silhouette: {avg_sil:.4f}")
        logger.info(f"  Robustness - ARI: {avg_noise_ari:.4f}, NMI: {avg_noise_nmi:.4f}")
        
        # Interpret stability
        if avg_ari > 0.8 and avg_nmi > 0.8:
            stability_rating = "Excellent"
        elif avg_ari > 0.6 and avg_nmi > 0.6:
            stability_rating = "Good"
        elif avg_ari > 0.4 and avg_nmi > 0.4:
            stability_rating = "Moderate"
        else:
            stability_rating = "Poor"
        
        # Interpret robustness
        if avg_noise_ari > 0.8 and avg_noise_nmi > 0.8:
            robustness_rating = "Excellent"
        elif avg_noise_ari > 0.6 and avg_noise_nmi > 0.6:
            robustness_rating = "Good"
        elif avg_noise_ari > 0.4 and avg_noise_nmi > 0.4:
            robustness_rating = "Moderate"
        else:
            robustness_rating = "Poor"
            
        logger.info(f"Stability rating: {stability_rating}")
        logger.info(f"Robustness rating: {robustness_rating}")
        
        # Plot stability metrics
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        x = np.arange(n_splits)
        width = 0.35
        plt.bar(x - width/2, [m[0] for m in stability_metrics], width, label='ARI')
        plt.bar(x + width/2, [m[1] for m in stability_metrics], width, label='NMI')
        plt.axhline(y=avg_ari, color='r', linestyle='--', alpha=0.5, label=f'Avg ARI: {avg_ari:.3f}')
        plt.axhline(y=avg_nmi, color='g', linestyle='--', alpha=0.5, label=f'Avg NMI: {avg_nmi:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.title('Cluster Stability Across Folds')
        plt.xticks(x, [f'Fold {i+1}' for i in range(n_splits)])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(x - width/2, [m[0] for m in robustness_metrics], width, label='Noise ARI')
        plt.bar(x + width/2, [m[1] for m in robustness_metrics], width, label='Noise NMI')
        plt.axhline(y=avg_noise_ari, color='r', linestyle='--', alpha=0.5, 
                   label=f'Avg ARI: {avg_noise_ari:.3f}')
        plt.axhline(y=avg_noise_nmi, color='g', linestyle='--', alpha=0.5, 
                   label=f'Avg NMI: {avg_noise_nmi:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.title('Noise Robustness Across Folds')
        plt.xticks(x, [f'Fold {i+1}' for i in range(n_splits)])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_validation.png'))
        plt.close()
        
        # Store and return validation results
        validation_results = {
            'stability_metrics': stability_metrics,
            'robustness_metrics': robustness_metrics,
            'avg_ari': avg_ari,
            'avg_nmi': avg_nmi,
            'avg_silhouette': avg_sil,
            'avg_noise_ari': avg_noise_ari,
            'avg_noise_nmi': avg_noise_nmi,
            'stability_rating': stability_rating,
            'robustness_rating': robustness_rating
        }
        
        return validation_results
    
    def feature_importance_analysis(self, 
                                  X: np.ndarray, 
                                  feature_names: List[str],
                                  cluster_labels: np.ndarray) -> Dict:
        """
        Perform feature importance analysis to identify which features most strongly
        influence the cluster assignments.
        
        Args:
            X: Original feature matrix
            feature_names: Names of the features
            cluster_labels: Cluster assignments
            
        Returns:
            Dictionary with feature importance results
        """
        logger.info("Analyzing feature importance...")
        
        # 1. ANOVA F-value for each feature
        from sklearn.feature_selection import f_classif
        
        f_values, p_values = f_classif(X, cluster_labels)
        feature_importance = [(name, f, p) for name, f, p in zip(feature_names, f_values, p_values)]
        sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
        
        # Calculate significant features (p < 0.05)
        significant_features = [name for name, _, p in feature_importance if p < 0.05]
        logger.info(f"Found {len(significant_features)} statistically significant features")
        
        # 2. Plot feature importance
        plt.figure(figsize=(12, 8))
        
        # Take top 15 features by F-value
        top_n = min(15, len(sorted_importance))
        top_features = sorted_importance[:top_n]
        
        # Create barplot
        names = [f[0] for f in top_features]
        f_vals = [f[1] for f in top_features]
        p_vals = [f[2] for f in top_features]
        
        # Color bars by significance
        colors = ['green' if p < 0.01 else 'yellowgreen' if p < 0.05 else 'orange' for p in p_vals]
        
        bars = plt.barh(range(len(names)), f_vals, color=colors)
        plt.yticks(range(len(names)), names)
        plt.xlabel('F-value')
        plt.title('Feature Importance (ANOVA F-value)')
        
        # Add p-value annotations
        for i, p in enumerate(p_vals):
            if p < 0.001:
                plt.text(f_vals[i] + max(f_vals)*0.02, i, "p < 0.001", va='center')
            elif p < 0.01:
                plt.text(f_vals[i] + max(f_vals)*0.02, i, "p < 0.01", va='center')
            elif p < 0.05:
                plt.text(f_vals[i] + max(f_vals)*0.02, i, "p < 0.05", va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance_anova.png'))
        plt.close()
        
        # 3. Feature distributions by cluster
        plt.figure(figsize=(15, 10))
        
        # Take top 6 significant features
        top_sig_features = [name for name, _, p in sorted_importance if p < 0.05][:6]
        
        if len(top_sig_features) < 6:
            # If fewer than 6 significant features, include some non-significant ones
            remaining = [name for name, _, p in sorted_importance if name not in top_sig_features]
            top_sig_features.extend(remaining[:6-len(top_sig_features)])
        
        # Create subplots for each feature
        for i, feature in enumerate(top_sig_features[:6]):
            plt.subplot(2, 3, i+1)
            
            # Get feature index
            feature_idx = feature_names.index(feature)
            
            # Create boxplot
            feature_data = []
            for cluster in np.unique(cluster_labels):
                cluster_values = X[cluster_labels == cluster, feature_idx]
                feature_data.append(cluster_values)
            
            plt.boxplot(feature_data, labels=[f'C{i}' for i in np.unique(cluster_labels)])
            plt.title(feature)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_distribution_by_cluster.png'))
        plt.close()
        
        # 4. Heatmap of cluster centroids
        # Calculate cluster centroids
        centroids = []
        for cluster in np.unique(cluster_labels):
            cluster_data = X[cluster_labels == cluster]
            centroids.append(np.mean(cluster_data, axis=0))
        
        centroid_matrix = np.vstack(centroids)
        
        # Scale centroids for better visualization
        from sklearn.preprocessing import MinMaxScaler
        centroid_scaler = MinMaxScaler()
        scaled_centroids = centroid_scaler.fit_transform(centroid_matrix)
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(scaled_centroids, annot=False, cmap='coolwarm', 
                   xticklabels=feature_names, yticklabels=[f'Cluster {i}' for i in np.unique(cluster_labels)])
        plt.title('Cluster Centroids')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_centroids_heatmap.png'))
        plt.close()
        
        # 5. Permutation importance (for classification task treating clusters as labels)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        
        # Train a classifier to predict cluster labels
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, cluster_labels)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(rf, X, cluster_labels, n_repeats=10, random_state=42)
        
        # Sort features by importance
        perm_importance_sorted = sorted(
            zip(feature_names, perm_importance.importances_mean),
            key=lambda x: x[1], reverse=True
        )
        
        # Plot permutation importance
        plt.figure(figsize=(12, 8))
        names = [f[0] for f in perm_importance_sorted[:15]]  # Top 15 features
        imp_vals = [f[1] for f in perm_importance_sorted[:15]]
        
        plt.barh(range(len(names)), imp_vals)
        plt.yticks(range(len(names)), names)
        plt.xlabel('Permutation Importance')
        plt.title('Feature Importance (Permutation)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance_permutation.png'))
        plt.close()
        
        # Return importance analysis results
        return {
            'anova_importance': sorted_importance,
            'significant_features': significant_features,
            'permutation_importance': perm_importance_sorted,
            'centroids': centroid_matrix,
            'feature_names': feature_names
        }

    def generate_advanced_visualizations(self, 
                                      df: pd.DataFrame,
                                      X: np.ndarray,
                                      cluster_labels: np.ndarray,
                                      feature_names: List[str],
                                      importance_results: Dict):
        """
        Generate advanced visualizations of the clustering results.
        
        Args:
            df: Original DataFrame
            X: Original feature matrix
            cluster_labels: Cluster assignments
            feature_names: Names of the features
            importance_results: Results from feature importance analysis
        """
        logger.info("Generating advanced visualizations...")
        
        # Add cluster labels to the original dataframe
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # 1. PCA visualization colored by cluster
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, 
                             cmap='viridis', s=100, alpha=0.8)
        plt.colorbar(scatter, label='Cluster')
        plt.title('PCA Visualization of Clusters')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.grid(True, alpha=0.3)
        
        # Add server names as annotations if available
        if 'server_name' in df.columns:
            for i, server_name in enumerate(df['server_name']):
                plt.annotate(server_name, (pca_result[i, 0], pca_result[i, 1]), 
                            fontsize=8, alpha=0.7)
        
        plt.savefig(os.path.join(self.output_dir, 'cluster_pca_visualization.png'))
        plt.close()
        
        # 2. t-SNE visualization colored by cluster
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(X)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, 
                             cmap='viridis', s=100, alpha=0.8)
        plt.colorbar(scatter, label='Cluster')
        plt.title('t-SNE Visualization of Clusters')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        
        # Add server names as annotations if available
        if 'server_name' in df.columns:
            for i, server_name in enumerate(df['server_name']):
                plt.annotate(server_name, (tsne_result[i, 0], tsne_result[i, 1]), 
                            fontsize=8, alpha=0.7)
        
        plt.savefig(os.path.join(self.output_dir, 'cluster_tsne_visualization.png'))
        plt.close()
        
        # 3. 3D visualization if we have at least 3 significant features
        sig_features = importance_results['significant_features']
        if len(sig_features) >= 3:
            # Use top 3 significant features
            top3_features = sig_features[:3]
            top3_indices = [feature_names.index(f) for f in top3_features]
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(X[:, top3_indices[0]], X[:, top3_indices[1]], X[:, top3_indices[2]], 
                               c=cluster_labels, cmap='viridis', s=100, alpha=0.8)
            
            ax.set_title('3D Visualization of Top 3 Significant Features')
            ax.set_xlabel(top3_features[0])
            ax.set_ylabel(top3_features[1])
            ax.set_zlabel(top3_features[2])
            
            plt.colorbar(scatter, label='Cluster')
            plt.savefig(os.path.join(self.output_dir, 'cluster_3d_visualization.png'))
            plt.close()
        
        # 4. Pair plot of top features
        top_features = [name for name, _, _ in importance_results['anova_importance'][:4]]
        
        if 'server_name' in df.columns:
            plot_df = df_with_clusters[top_features + ['cluster', 'server_name']]
        else:
            plot_df = df_with_clusters[top_features + ['cluster']]
        
        try:
            plt.figure(figsize=(12, 10))
            sns.pairplot(plot_df, hue='cluster', palette='viridis', diag_kind='kde',
                        plot_kws={'alpha': 0.6, 's': 80})
            plt.suptitle('Pair Plot of Top Features', y=1.02)
            plt.savefig(os.path.join(self.output_dir, 'cluster_pair_plot.png'))
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create pair plot: {e}")
        
        # 5. Radar chart of cluster profiles
        try:
            # Calculate cluster centroids
            centroids = importance_results['centroids']
            
            # Use top significant features (max 8 for readability)
            top_sig_features = importance_results['significant_features'][:8]
            top_sig_indices = [feature_names.index(f) for f in top_sig_features]
            
            # Create radar chart (spider plot)
            from matplotlib.path import Path
            from matplotlib.spines import Spine
            from matplotlib.transforms import Affine2D
            
            def radar_factory(num_vars, frame='circle'):
                # Create the radar chart
                theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
                
                class RadarAxes(plt.Axes):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.set_theta_zero_location('N')
                    
                    def fill(self, *args, closed=True, **kwargs):
                        return super().fill(self.theta, *args, closed=closed, **kwargs)
                    
                    def plot(self, *args, **kwargs):
                        return super().plot(self.theta, *args, **kwargs)
                    
                    def set_varlabels(self, labels):
                        self.set_thetagrids(np.degrees(theta), labels)
                    
                    def _gen_axes_patch(self):
                        if frame == 'circle':
                            return plt.Circle((0.5, 0.5), 0.5)
                        else:
                            return plt.RegularPolygon(
                                (0.5, 0.5), num_vars, radius=0.5, orientation=np.pi/2)
                    
                    def _gen_axes_spines(self):
                        if frame == 'circle':
                            return {}
                        spine_dict = {}
                        spine_path = Path.unit_regular_polygon(num_vars)
                        spine_tr = Affine2D().scale(.5).translate(.5, .5)
                        for i in range(num_vars):
                            spine = Spine(self, 'polar', spine_tr.transform_path(spine_path))
                            spine.set_transform(
                                plt.Affine2D().rotate(theta[i]).translate(0.5, 0.5) + self.transAxes)
                            spine_dict[f"spine_{i}"] = spine
                        return spine_dict
                
                register_projection(RadarAxes)
                return theta
            
            # Normalize the centroids for radar chart
            centroid_subset = centroids[:, top_sig_indices]
            min_vals = np.min(centroid_subset, axis=0)
            max_vals = np.max(centroid_subset, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # Avoid division by zero
            
            normalized_centroids = (centroid_subset - min_vals) / range_vals
            
            # Create the radar chart
            theta = radar_factory(len(top_sig_features), frame='polygon')
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))
            colors = plt.cm.viridis(np.linspace(0, 1, len(centroids)))
            
            for i, (centroid, color) in enumerate(zip(normalized_centroids, colors)):
                ax.plot(theta, centroid, color=color, label=f'Cluster {i}')
                ax.fill(theta, centroid, color=color, alpha=0.1)
            
            ax.set_varlabels(top_sig_features)
            plt.legend(loc='upper right')
            plt.title('Radar Chart of Cluster Profiles')
            
            plt.savefig(os.path.join(self.output_dir, 'cluster_radar_chart.png'))
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create radar chart: {e}")

    def generate_descriptive_cluster_labels(self, 
                                       df: pd.DataFrame, 
                                       cluster_labels: np.ndarray,
                                       feature_names: List[str],
                                       importance_results: Dict) -> pd.DataFrame:
        """
        Generate descriptive labels for each cluster based on their key characteristics.
        
        Args:
            df: Original DataFrame
            cluster_labels: Cluster assignments
            feature_names: Feature names
            importance_results: Results from feature importance analysis
            
        Returns:
            DataFrame with cluster labels and descriptions
        """
        logger.info("Generating descriptive cluster labels...")
        
        # Add cluster labels to the original data
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # Calculate cluster means for numeric features
        numeric_cols = df.select_dtypes(include=['number']).columns
        valid_feature_names = [f for f in feature_names if f in numeric_cols]
        
        cluster_means = df_with_clusters.groupby('cluster')[valid_feature_names].mean()
        
        # Calculate overall means for comparison
        overall_means = df[valid_feature_names].mean()
        
        # Calculate z-scores for each cluster (how far from overall mean)
        z_scores = (cluster_means - overall_means) / df[valid_feature_names].std()
        
        # Get significant features for each cluster (from importance analysis)
        significant_features = importance_results['significant_features']
        
        # Generate descriptive labels based on z-scores of significant features
        descriptive_labels = []
        
        for cluster_id in np.unique(cluster_labels):
            # Get cluster z-scores
            cluster_z = z_scores.loc[cluster_id]
            
            # Filter to significant features
            sig_features_z = [(feature, cluster_z[feature]) 
                             for feature in significant_features 
                             if feature in cluster_z.index]
            
            # Sort by absolute z-score (most distinctive first)
            sorted_features = sorted(sig_features_z, key=lambda x: abs(x[1]), reverse=True)
            
            # Get top 3 most distinctive features (or fewer if not available)
            top_n = min(3, len(sorted_features))
            top_features = sorted_features[:top_n]
            
            # Create label components
            label_parts = []
            
            for feature, z_value in top_features:
                actual_value = cluster_means.loc[cluster_id, feature]
                
                # Determine descriptor based on z-score
                if z_value > 1.5:
                    descriptor = "very high"
                elif z_value > 0.7:
                    descriptor = "high"
                elif z_value < -1.5:
                    descriptor = "very low"
                elif z_value < -0.7:
                    descriptor = "low"
                else:
                    descriptor = "average"
                
                # Format the feature name for readability
                formatted_feature = feature.replace('_', ' ')
                
                # Add actual value (formatted appropriately)
                if 'cost' in feature or 'estimated' in feature:
                    value_str = f"${actual_value:.2f}"
                elif 'utilization' in feature:
                    value_str = f"{actual_value:.1f}%"
                elif 'count' in feature or 'cores' in feature or 'projects' in feature:
                    value_str = f"{actual_value:.1f}"
                elif 'gb' in feature:
                    value_str = f"{actual_value:.1f}GB"
                else:
                    value_str = f"{actual_value:.2f}"
                
                label_parts.append(f"{descriptor} {formatted_feature} ({value_str})")
            
            # Create full descriptive label
            descriptive_label = " | ".join(label_parts)
            
            # Add size information
            cluster_size = np.sum(cluster_labels == cluster_id)
            size_percentage = (cluster_size / len(df)) * 100
            
            # Get server examples if available
            server_examples = ""
            if 'server_name' in df.columns:
                servers = df_with_clusters[df_with_clusters['cluster'] == cluster_id]['server_name']
                server_examples = ", ".join(servers.head(3).astype(str))
            
            descriptive_labels.append({
                'cluster': cluster_id,
                'descriptive_label': descriptive_label,
                'size': cluster_size,
                'percentage': size_percentage,
                'server_examples': server_examples
            })
        
        # Convert to DataFrame
        labels_df = pd.DataFrame(descriptive_labels)
        
        # Export descriptive labels
        labels_df.to_csv(os.path.join(self.output_dir, 'descriptive_cluster_labels.csv'), index=False)
        
        # Print the descriptive labels
        logger.info("\nDescriptive Cluster Labels:")
        for _, row in labels_df.iterrows():
            logger.info(f"Cluster {row['cluster']} ({row['size']} servers, {row['percentage']:.1f}%): {row['descriptive_label']}")
            if row['server_examples']:
                logger.info(f"  Examples: {row['server_examples']}")
        
        return labels_df

    def generate_optimization_recommendations(self, 
                                       df: pd.DataFrame,
                                       cluster_labels: np.ndarray,
                                       descriptive_labels: pd.DataFrame) -> pd.DataFrame:
        """
        Generate actionable recommendations for server optimization based on cluster analysis.
        
        Args:
            df: Original DataFrame
            cluster_labels: Cluster assignments
            descriptive_labels: DataFrame with descriptive cluster labels
            
        Returns:
            DataFrame with optimization recommendations
        """
        logger.info("Generating optimization recommendations...")
        
        # Add cluster labels to the original data
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # Debug: Print dataframe info
        logger.info(f"DataFrame shape: {df_with_clusters.shape}")
        logger.info(f"DataFrame columns: {df_with_clusters.columns.tolist()}")
        logger.info(f"Number of clusters: {len(np.unique(cluster_labels))}")
        
        # Define thresholds for recommendations
        thresholds = {
            'high_cpu_utilization': 70,  # CPU utilization above 70% is high
            'low_cpu_utilization': 30,   # CPU utilization below 30% is low
            'high_memory_utilization': 70,  # Memory utilization above 70% is high
            'low_memory_utilization': 30,   # Memory utilization below 30% is low
            'low_project_density': 1.5,   # Fewer than 1.5 projects per core is low density
            'high_cost_per_project': df['cost_per_project'].quantile(0.7) if 'cost_per_project' in df.columns else 150
        }
        
        # Generate recommendations for each cluster
        recommendations = []
        
        for cluster_id in np.unique(cluster_labels):
            logger.info(f"Processing cluster {cluster_id}...")
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            # Get descriptive label
            desc_label = descriptive_labels[descriptive_labels['cluster'] == cluster_id]['descriptive_label'].values[0]
            
            # Get server examples if available
            server_examples = ""
            if 'server_name' in df.columns:
                servers = cluster_data['server_name'].head(3).astype(str)
                server_examples = ", ".join(servers)
            
            # Initialize recommendation components
            resource_recommendations = []
            cost_recommendations = []
            performance_recommendations = []
            
            # Check for resource optimization opportunities based on utilization
            if 'cpu_utilization' in cluster_data.columns:
                cpu_util = cluster_data['cpu_utilization'].mean()
                logger.info(f"  Cluster {cluster_id} - Average CPU utilization: {cpu_util:.1f}%")
                
                if cpu_util < thresholds['low_cpu_utilization']:
                    resource_recommendations.append(
                        f"Consider reducing CPU allocation as utilization is low ({cpu_util:.1f}%)"
                    )
                elif cpu_util > thresholds['high_cpu_utilization']:
                    performance_recommendations.append(
                        f"Monitor for CPU-bound performance issues as utilization is high ({cpu_util:.1f}%)"
                    )
                    resource_recommendations.append(
                        "Consider increasing CPU allocation to prevent performance bottlenecks"
                    )
            
            if 'memory_utilization' in cluster_data.columns:
                mem_util = cluster_data['memory_utilization'].mean()
                logger.info(f"  Cluster {cluster_id} - Average memory utilization: {mem_util:.1f}%")
                
                if mem_util < thresholds['low_memory_utilization']:
                    resource_recommendations.append(
                        f"Consider reducing memory allocation as utilization is low ({mem_util:.1f}%)"
                    )
                elif mem_util > thresholds['high_memory_utilization']:
                    performance_recommendations.append(
                        f"Monitor for memory-bound performance issues as utilization is high ({mem_util:.1f}%)"
                    )
                    resource_recommendations.append(
                        "Consider increasing memory allocation to prevent performance bottlenecks"
                    )
            
            # Check for cost optimization opportunities
            if 'cost_per_project' in cluster_data.columns and 'project_count' in cluster_data.columns:
                cost_per_project = cluster_data['cost_per_project'].mean()
                project_count = cluster_data['project_count'].mean()
                logger.info(f"  Cluster {cluster_id} - Average cost per project: ${cost_per_project:.2f}, Project count: {project_count:.1f}")
                
                if cost_per_project > thresholds['high_cost_per_project'] and project_count > 0:
                    cost_recommendations.append(
                        f"High cost per project (${cost_per_project:.2f}). Consider resource optimization or consolidation"
                    )
                
                if 'projects_per_core' in cluster_data.columns:
                    projects_per_core = cluster_data['projects_per_core'].mean()
                    logger.info(f"  Cluster {cluster_id} - Average projects per core: {projects_per_core:.1f}")
                    
                    if projects_per_core < thresholds['low_project_density'] and project_count > 1:
                        cost_recommendations.append(
                            f"Low project density ({projects_per_core:.1f} projects per core). Consider consolidating projects to fewer servers"
                        )
            
            # Check for idle or underutilized servers
            if 'project_count' in cluster_data.columns:
                project_count = cluster_data['project_count'].mean()
                
                if project_count < 1:
                    resource_recommendations.append(
                        "Server appears to be idle (no projects). Consider decommissioning or repurposing"
                    )
                elif project_count < 3 and 'cpu_utilization' in cluster_data.columns:
                    cpu_util = cluster_data['cpu_utilization'].mean()
                    if cpu_util < thresholds['low_cpu_utilization']:
                        resource_recommendations.append(
                            f"Server is underutilized (few projects: {project_count:.1f}, low CPU: {cpu_util:.1f}%). Consider consolidation"
                        )
            
            # Combine recommendations
            all_recommendations = resource_recommendations + cost_recommendations + performance_recommendations
            logger.info(f"  Cluster {cluster_id} - Generated {len(all_recommendations)} recommendations")
            
            # Add to recommendations list
            recommendations.append({
                'cluster': cluster_id,
                'descriptive_label': desc_label,
                'server_examples': server_examples,
                'recommendations': all_recommendations
            })
        
        # Convert to DataFrame
        recommendations_df = pd.DataFrame(recommendations)
        
        # Export recommendations
        output_path = os.path.join(self.output_dir, 'optimization_recommendations.csv')
        recommendations_df.to_csv(output_path, index=False)
        logger.info(f"Saved recommendations to {output_path}")
        
        # Print the recommendations
        logger.info("\nOptimization Recommendations:")
        for _, row in recommendations_df.iterrows():
            logger.info(f"Cluster {row['cluster']}: {row['descriptive_label']}")
            if row['server_examples']:
                logger.info(f"  Examples: {row['server_examples']}")
            for i, rec in enumerate(row['recommendations']):
                logger.info(f"  {i+1}. {rec}")
        
        return recommendations_df

    def run_enhanced_analysis(self, 
                           data_path: Optional[str] = None,
                           dim_reduction_methods: List[str] = ['pca', 'vae'],
                           latent_dim: int = 8,
                           handle_outliers: bool = True,
                           feature_selection: str = 'kbest',
                           max_clusters: int = 12,
                           min_clusters: int = 2) -> Dict:
        """
        Run the complete enhanced clustering analysis pipeline.
        
        Args:
            data_path: Path to CSV file with data (optional)
            dim_reduction_methods: List of dimensionality reduction methods to use
            latent_dim: Dimension of the latent space
            handle_outliers: Whether to detect and handle outliers
            feature_selection: Feature selection method
            max_clusters: Maximum number of clusters to try
            min_clusters: Minimum number of clusters to try
            
        Returns:
            Dictionary with analysis results
        """
        # Load data
        df = self.load_data(filepath=data_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Preprocess data
        preproc_results = self.preprocess_data(
            df, 
            scaler_type='robust', 
            handle_outliers=handle_outliers, 
            feature_selection=feature_selection
        )
        
        # Train dimensionality reduction models
        dim_reduction_results = self.train_dimensionality_reduction_models(
            preproc_results['selected_data'], 
            methods=dim_reduction_methods, 
            latent_dim=latent_dim
        )
        
        # Perform clustering
        clustering_results = self.perform_ensemble_clustering(
            dim_reduction_results,
            max_clusters=max_clusters,
            min_clusters=min_clusters
        )
        
        # Get best clustering result
        best_result = clustering_results['best_result']
        best_labels = best_result['cluster_labels']
        
        # Validate clusters
        validation_results = self.validate_clusters(
            preproc_results['selected_data'], 
            best_result
        )
        
        # Analyze feature importance
        importance_results = self.feature_importance_analysis(
            preproc_results['selected_data'], 
            preproc_results['selected_feature_names'], 
            best_labels
        )
        
        # Generate visualizations
        self.generate_advanced_visualizations(
            df, 
            preproc_results['selected_data'], 
            best_labels, 
            preproc_results['selected_feature_names'], 
            importance_results
        )
        
        # Generate descriptive labels
        descriptive_labels = self.generate_descriptive_cluster_labels(
            df, 
            best_labels, 
            preproc_results['selected_feature_names'], 
            importance_results
        )
        
        # Generate optimization recommendations
        recommendations = self.generate_optimization_recommendations(
            df, 
            best_labels, 
            descriptive_labels
        )
        
        # Combine all results
        results = {
            **best_result,
            'validation_results': validation_results,
            'importance_results': importance_results,
            'descriptive_labels': descriptive_labels,
            'recommendations': recommendations,
            'preprocessing': preproc_results
        }
        
        logger.info(f"Enhanced analysis complete! Results saved to {self.output_dir}")
        
        return results

# Add a main function to run the clustering process
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run enhanced clustering analysis on server metrics')
    parser.add_argument('--input', type=str, help='Path to input CSV file')
    parser.add_argument('--output', type=str, help='Path to output directory')
    parser.add_argument('--db_url', type=str, help='Database connection URL')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output if args.output else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the clustering model
    logger.info("Initializing DeepClusteringEnsemble...")
    model = DeepClusteringEnsemble(output_dir=output_dir, db_url=args.db_url)
    
    try:
        # Load data
        logger.info("Loading data...")
        df = model.load_data(filepath=args.input)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Preprocess data
        logger.info("Preprocessing data...")
        preproc_results = model.preprocess_data(df, scaler_type='robust', handle_outliers=True, feature_selection='kbest')
        
        # Train dimensionality reduction models
        logger.info("Training dimensionality reduction models...")
        dim_reduction_results = model.train_dimensionality_reduction_models(
            preproc_results['selected_data'], 
            methods=['pca', 'vae'], 
            latent_dim=8
        )
        
        # Perform clustering
        logger.info("Performing ensemble clustering...")
        clustering_results = model.perform_ensemble_clustering(dim_reduction_results)
        
        # Get best clustering result
        best_result = clustering_results['best_result']
        best_labels = best_result['cluster_labels']
        
        # Validate clusters
        logger.info("Validating clusters...")
        validation_results = model.validate_clusters(preproc_results['selected_data'], best_result)
        
        # Analyze feature importance
        logger.info("Analyzing feature importance...")
        importance_results = model.feature_importance_analysis(
            preproc_results['selected_data'], 
            preproc_results['selected_feature_names'], 
            best_labels
        )
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        model.generate_advanced_visualizations(
            df, 
            preproc_results['selected_data'], 
            best_labels, 
            preproc_results['selected_feature_names'], 
            importance_results
        )
        
        # Generate descriptive labels
        logger.info("Generating descriptive cluster labels...")
        descriptive_labels = model.generate_descriptive_cluster_labels(
            df, 
            best_labels, 
            preproc_results['selected_feature_names'], 
            importance_results
        )
        
        # Generate optimization recommendations
        logger.info("Generating optimization recommendations...")
        recommendations = model.generate_optimization_recommendations(
            df, 
            best_labels, 
            descriptive_labels
        )
        
        logger.info(f"Analysis complete! Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during clustering analysis: {e}", exc_info=True)
        raise