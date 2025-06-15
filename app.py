import streamlit as st
import pandas as pd
import numpy as np
import json
import umap
import hdbscan
from datetime import datetime
import plotly.express as px
from typing import Dict, List, Tuple
import os
from pathlib import Path
import openai
from openai import OpenAI
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
# OPENAI_API_KEY is now read from environment variable in main()
EMBEDDING_DIM = 50  # Dimension for UMAP embedding
MIN_CLUSTER_SIZE = 2  # Minimum size for HDBSCAN clusters

# Remove global openai_client initialization
# openai_client = OpenAI(api_key=OPENAI_API_KEY)

@dataclass
class UserBehavior:
    user_id: str
    timestamp: datetime
    action: str
    metadata: Dict

class BehavioralAnalyzer:
    def __init__(self, api_key: str):
        self.clusterer = None
        self.embeddings = None
        self.user_behaviors = None
        self.cluster_summaries = None
        # Initialize OpenAI client within the class
        if not api_key:
            raise ValueError("OpenAI API Key not provided or found in environment variables.")
        # Use OpenRouter base URL
        self.openai_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        
    def process_csv(self, csv_path: str) -> pd.DataFrame:
        """Process CSV file into a DataFrame with proper types."""
        try:
            # Read CSV with proper handling of quoted fields
            df = pd.read_csv(csv_path, quoting=1)  # QUOTE_ALL mode
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Drop rows with missing action or timestamp before processing
            df.dropna(subset=['action', 'timestamp'], inplace=True)
            
            # Process metadata column
            def parse_metadata(x):
                try:
                    if isinstance(x, str) and x.strip():  # Check if string and not empty/whitespace
                        # Remove any outer quotes and parse JSON
                        x = x.strip('"')
                        return json.loads(x)
                    # If not a string, or empty string, return empty dict or existing dict
                    return x if isinstance(x, dict) else {}
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing metadata: {str(e)} for value: '{x}'")
                    return {}
                except Exception as e:
                    logger.warning(f"Unexpected error parsing metadata: {str(e)} for value: '{x}'")
                    return {}
            
            df['metadata'] = df['metadata'].apply(parse_metadata)
            
            logger.info(f"Successfully processed CSV with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise ValueError(f"Failed to process CSV file: {str(e)}")
    
    def create_behavior_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Create a behavior matrix for clustering and return user IDs."""
        # Get unique actions
        actions = df['action'].unique()
        
        # Create user-action matrix
        user_actions = pd.crosstab(df['user_id'], df['action'])
        
        # Fill missing actions with 0
        for action in actions:
            if action not in user_actions.columns:
                user_actions[action] = 0
        
        return user_actions.values, user_actions.index.tolist()
    
    def cluster_users(self, behavior_matrix: np.ndarray) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
        """Cluster users using UMAP + HDBSCAN."""
        # Reduce dimensionality with UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, spread=0.5, min_dist=0.1)
        embeddings = reducer.fit_transform(behavior_matrix)
        
        # Cluster with HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE)
        cluster_labels = clusterer.fit_predict(behavior_matrix)
        
        return embeddings, clusterer
    
    def get_cluster_summary(self, df: pd.DataFrame, cluster_id: int) -> Dict:
        """Generate summary statistics for a cluster."""
        cluster_users = df[df['cluster'] == cluster_id]['user_id'].unique()
        cluster_data = df[df['user_id'].isin(cluster_users)]
        
        # Initialize detailed summaries
        feature_usage_details = {}
        feedback_comments = []
        proposal_status = {'created': 0, 'revised': 0, 'closed': 0, 'viewed_only': 0}
        
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
            elif action == 'proposal_create':
                proposal_status['created'] += 1
            elif action == 'proposal_revision':
                proposal_status['revised'] += 1
            elif action == 'proposal_close':
                proposal_status['closed'] += 1
            elif action == 'proposal_view' and action not in cluster_data[cluster_data['user_id'] == row['user_id']]['action'].values:
                # This is a simplification; a more robust check would involve checking for subsequent close actions
                proposal_status['viewed_only'] += 1

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
            'proposal_flow_status': proposal_status
        }
        
        return stats
    
    def get_llm_insights(self, cluster_summary: Dict) -> Dict:
        """Get pain points and recommendations from LLM."""
        # Construct detailed statistics string
        detailed_stats = """
        Detailed Metrics:
        - Specific Feature Usage: {cluster_summary['feature_usage_details']}
        - User Feedback Comments: {cluster_summary['feedback_comments']}
        - Proposal Flow Status (Created/Revised/Closed/Viewed Only): {cluster_summary['proposal_flow_status']}
        """

        prompt = f"""Analyze this user cluster behavior and provide highly specific and actionable insights:
        Cluster Statistics:
        - Number of users: {cluster_summary['user_count']}
        - Total actions: {cluster_summary['total_actions']}
        - Average actions per user: {cluster_summary['avg_actions_per_user']}
        - Most common actions: {cluster_summary['common_actions']}
        - Time range: {cluster_summary['time_range']['start']} to {cluster_summary['time_range']['end']}
        {detailed_stats}

        Based on the provided data, please provide:
        1. **Top 3 Pain Points:** Be extremely specific. Mention *which features*, *which steps in a process*, or *which UI elements* are problematic. Provide concrete examples where possible (e.g., "Users struggle with the 'Add new client' button on the dashboard because it's not clearly visible.").
        2. **Top 3 Product Recommendations:** Be highly actionable. Suggest *specific features to test out*, *UI/UX changes* (e.g., "Change the color of the 'Submit Proposal' button to green"), or *onboarding flow experiments*. Each recommendation should be a clear, testable hypothesis.
        
        Format the response as a JSON with 'pain_points' and 'recommendations' arrays."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            insights = json.loads(response.choices[0].message.content)
            return insights
            
        except Exception as e:
            logger.error(f"Error getting LLM insights: {str(e)}")
            return {
                'pain_points': ["Error getting insights"],
                'recommendations': ["Error getting recommendations"]
            }

def main():
    st.set_page_config(page_title="Grupa.io Behavioral Analysis", layout="wide")
    
    # Initialize session state
    if 'roadmap' not in st.session_state:
        st.session_state.roadmap = []
    
    # Sidebar
    st.sidebar.title("Grupa.io Analysis")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload User Behavior CSV", type=['csv'])
    
    if uploaded_file is not None:
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable.")
            return

        # Initialize analyzer
        analyzer = BehavioralAnalyzer(api_key)
        
        try:
            # Process CSV
            df = analyzer.process_csv(uploaded_file)
            
            # Create behavior matrix
            behavior_matrix, user_ids_for_clustering = analyzer.create_behavior_matrix(df)
            
            # Cluster users
            embeddings, clusterer = analyzer.cluster_users(behavior_matrix)
            
            # Create a DataFrame for user_id and cluster labels from the clustering input
            user_to_cluster = pd.DataFrame({
                'user_id': user_ids_for_clustering,
                'cluster': clusterer.labels_
            })
            
            # Merge cluster labels back to the original DataFrame
            df = df.merge(user_to_cluster, on='user_id', how='left')
            
            # Create scatter plot
            # The embeddings are ordered by user_ids_for_clustering. Ensure color mapping is correct.
            # We need to create a temporary DataFrame for plotly that has the user_ids and their embeddings.
            plot_df = pd.DataFrame({
                'user_id': user_ids_for_clustering,
                'x': embeddings[:, 0],
                'y': embeddings[:, 1],
                'cluster': clusterer.labels_ # Keep numerical for internal logic if needed
            })

            # Create a more descriptive cluster label for plotting
            plot_df['cluster_label'] = plot_df['cluster'].astype(str).replace({'-1': 'Noise'})
            plot_df['cluster_label'] = plot_df['cluster_label'].apply(lambda x: f'Cluster {x}' if x != 'Noise' else x)

            fig = px.scatter(
                plot_df,
                x='x',
                y='y',
                color='cluster_label', # Use the new descriptive label for coloring
                symbol='cluster_label', # Use the new descriptive label for symbols
                hover_data={'user_id': True, 'cluster': True, 'cluster_label': True},
                title="User Clusters"
            )
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Handle cluster selection
            selected_cluster_label = st.selectbox(
                "Select Cluster",
                options=sorted(plot_df['cluster_label'].unique()), # Use descriptive labels as options
                format_func=lambda x: x # The labels are already descriptive
            )
            
            # Map the selected label back to the numerical cluster ID for filtering
            selected_cluster_id = plot_df[plot_df['cluster_label'] == selected_cluster_label]['cluster'].iloc[0] if selected_cluster_label != 'Noise' else -1

            if selected_cluster_id != -1:
                # Get cluster summary
                cluster_summary = analyzer.get_cluster_summary(df, selected_cluster_id)
                
                # Get LLM insights
                insights = analyzer.get_llm_insights(cluster_summary)
                
                # Display cluster information
                st.subheader(f"Cluster {selected_cluster_label} Summary") # Use descriptive label for subheader
                with st.expander("View Raw Cluster Summary"): # Make cluster summary expandable
                    st.json(cluster_summary)
                
                # Display pain points
                st.subheader("Top Pain Points") # More descriptive subheader
                for point in insights['pain_points']:
                    st.markdown(f"- **{point}**") # Use markdown for bold bullet points
                
                # Display recommendations
                st.subheader("Top Product Recommendations") # More descriptive subheader
                for i, rec in enumerate(insights['recommendations']):
                    # Remove the Add to Roadmap button
                    st.markdown(f"{i+1}. **{rec}**") # Use markdown for bold bullet points
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 