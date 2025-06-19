import pandas as pd
import numpy as np
import umap
import hdbscan
from typing import Dict, List, Tuple

# Constants
EMBEDDING_DIM = 50  # Dimension for UMAP embedding
MIN_CLUSTER_SIZE = 2  # Minimum size for HDBSCAN clusters

class ClusteringAnalyzer:
    def __init__(self):
        self.clusterer = None
        self.embeddings = None
        self.user_behaviors = None
        self.cluster_summaries = None
        self.user_actions_df = None
        
    def create_behavior_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        """Create a behavior matrix for clustering and return user IDs and the full user-action DataFrame."""
        # Get unique actions
        actions = df['action'].unique()
        
        # Create user-action matrix
        user_actions_df = pd.crosstab(df['user_id'], df['action'])
        
        # Fill missing actions with 0
        for action in actions:
            if action not in user_actions_df.columns:
                user_actions_df[action] = 0
        
        return user_actions_df.values, user_actions_df.index.tolist(), user_actions_df
    
    def cluster_users(self, behavior_matrix: np.ndarray) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
        """Cluster users using UMAP + HDBSCAN."""
        # Reduce dimensionality with UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, spread=0.5, min_dist=0.001)
        embeddings = reducer.fit_transform(behavior_matrix)
        
        # Cluster with HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE)
        cluster_labels = clusterer.fit_predict(embeddings)
        
        return embeddings, clusterer
    
    def get_cluster_feature_importance(self, user_actions_df: pd.DataFrame, cluster_id: int, df: pd.DataFrame) -> Dict:
        """Identify top behavioral features differentiating a cluster."""
        # Get user IDs for the current cluster
        cluster_user_ids = df[df['cluster'] == cluster_id]['user_id'].unique()
        
        # Filter the user_actions_df for this cluster
        cluster_user_actions = user_actions_df.loc[cluster_user_ids]
        
        # Calculate mean action frequency for the cluster
        cluster_mean_actions = cluster_user_actions.mean()
        
        # Calculate overall mean action frequency
        overall_mean_actions = user_actions_df.mean()
        
        # Calculate the ratio of cluster mean to overall mean (avoid division by zero)
        # Add a small epsilon to avoid division by zero for overall_mean_actions that are 0
        ratio = cluster_mean_actions / (overall_mean_actions + 1e-6)
        
        # Identify top positive and negative differentiating features
        # Positive: actions much more frequent in this cluster
        # Negative: actions much less frequent in this cluster (compared to other actions)

        # We'll use a threshold, e.g., ratio > 1.5 for more frequent, ratio < 0.5 for less frequent
        differentiating_features = {
            'highly_frequent': ratio[ratio > 1.5].sort_values(ascending=False).index.tolist(),
            'less_frequent': ratio[ratio < 0.5].sort_values(ascending=True).index.tolist()
        }

        return differentiating_features

    def get_cluster_summary(self, df: pd.DataFrame, cluster_id: int) -> Dict:
        """Generate summary statistics for a cluster."""
        cluster_users = df[df['cluster'] == cluster_id]['user_id'].unique()
        cluster_data = df[df['user_id'].isin(cluster_users)]
        
        # Initialize detailed summaries
        feature_usage_details = {}
        feedback_comments = []
        shopping_status = {
            'product_view': 0,
            'search': 0,
            'filter': 0,
            'add_to_cart': 0,
            'remove_from_cart': 0,
            'abandon_cart': 0,
            'checkout_start': 0,
            'purchase_complete': 0
        }
        
        # Iterate through cluster data to gather detailed info
        for _, row in cluster_data.iterrows():
            action = row['action']
            metadata = row['metadata']
            
            if action == 'feature_use':
                feature_name = metadata.get('feature_name')
                if feature_name:
                    feature_usage_details[feature_name] = feature_usage_details.get(feature_name, 0) + 1
            elif action == 'feedback':
                comment = metadata.get('comment')
                if comment:
                    feedback_comments.append(comment)
            elif action in shopping_status:
                shopping_status[action] += 1

        # Calculate basic stats
        stats = {
            'user_count': len(cluster_users),
            'total_actions': len(cluster_data),
            'avg_actions_per_user': len(cluster_data) / len(cluster_users),
            'common_actions': cluster_data['action'].value_counts().head(5).to_dict(),
            'time_range': {
                'start': cluster_data['timestamp'].min().isoformat(),
                'end': cluster_data['timestamp'].max().isoformat()
            },
            'feature_usage_details': feature_usage_details,
            'feedback_comments': feedback_comments,
            'shopping_flow_status': shopping_status,
            'differentiating_features': self.get_cluster_feature_importance(self.user_actions_df, cluster_id, df)
        }
        
        # Calculate conversion and completion rates based on shopping flow
        # Conversion Rate: Users who completed a purchase / Users who started checkout
        started_checkout = shopping_status.get('checkout_start', 0)
        completed_purchase = shopping_status.get('purchase_complete', 0)
        stats['conversion_rate'] = (completed_purchase / started_checkout) * 100 if started_checkout > 0 else 0
        
        # Completion Rate: Users who completed a purchase / users who added to cart
        added_to_cart = shopping_status.get('add_to_cart', 0)
        stats['completion_rate'] = (completed_purchase / added_to_cart) * 100 if added_to_cart > 0 else 0

        # Calculate funnel metrics: number of unique users at each stage
        funnel_metrics = {
            'product_view': len(cluster_data[cluster_data['action'] == 'product_view']['user_id'].unique()),
            'search': len(cluster_data[cluster_data['action'] == 'search']['user_id'].unique()),
            'filter': len(cluster_data[cluster_data['action'] == 'filter']['user_id'].unique()),
            'add_to_cart': len(cluster_data[cluster_data['action'] == 'add_to_cart']['user_id'].unique()),
            'remove_from_cart': len(cluster_data[cluster_data['action'] == 'remove_from_cart']['user_id'].unique()),
            'abandon_cart': len(cluster_data[cluster_data['action'] == 'abandon_cart']['user_id'].unique()),
            'checkout_start': len(cluster_data[cluster_data['action'] == 'checkout_start']['user_id'].unique()),
            'purchase_complete': len(cluster_data[cluster_data['action'] == 'purchase_complete']['user_id'].unique())
        }
        stats['funnel_metrics'] = funnel_metrics

        return stats 