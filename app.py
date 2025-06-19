import streamlit as st
import pandas as pd
import numpy as np
import json
# import umap # Removed, now in clustering.py
# import hdbscan # Removed, now in clustering.py
from datetime import datetime
import plotly.express as px
from typing import Dict, List, Tuple
import os
from pathlib import Path
# import openai # Removed, now in llm_agent.py
# from openai import OpenAI # Removed, now in llm_agent.py
# from dataclasses import dataclass # Removed, now in utils.py
import logging

# Import modules
from llm_agent import LLMAgent
from clustering import ClusteringAnalyzer
from utils import process_csv, generate_markdown_report, simulate_ab_test

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
# OPENAI_API_KEY is now read from environment variable in main()
# EMBEDDING_DIM = 50  # Removed, now in clustering.py
# MIN_CLUSTER_SIZE = 2  # Removed, now in clustering.py

# Remove global openai_client initialization
# openai_client = OpenAI(api_key=OPENAI_API_KEY)

# @dataclass
# class UserBehavior:
#     user_id: str
#     timestamp: datetime
#     action: str
#     metadata: Dict

# class BehavioralAnalyzer:
#     def __init__(self, api_key: str):
#         self.clusterer = None
#         self.embeddings = None
#         self.user_behaviors = None
#         self.cluster_summaries = None
#         self.user_actions_df = None
#         # Initialize OpenAI client within the class
#         if not api_key:
#             raise ValueError("OpenAI API Key not provided or found in environment variables.")
#         # Use OpenRouter base URL
#         self.openai_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        
#     def process_csv(self, csv_path: str) -> pd.DataFrame:
#         """Process CSV file into a DataFrame with proper types."""
#         try:
#             # Read CSV with proper handling of quoted fields
#             df = pd.read_csv(csv_path, quoting=1)  # QUOTE_ALL mode
            
#             # Convert timestamp
#             df['timestamp'] = pd.to_datetime(df['timestamp'])
            
#             # Drop rows with missing action or timestamp before processing
#             df.dropna(subset=['action', 'timestamp'], inplace=True)
            
#             # Process metadata column
#             def parse_metadata(x):
#                 try:
#                     if isinstance(x, str) and x.strip():  # Check if string and not empty/whitespace
#                         # Remove any outer quotes and parse JSON
#                         x = x.strip('"')
#                         return json.loads(x)
#                     # If not a string, or empty string, return empty dict or existing dict
#                     return x if isinstance(x, dict) else {}
#                 except json.JSONDecodeError as e:
#                     logger.warning(f"Error parsing metadata: {str(e)} for value: '{x}'")
#                     return {}
#                 except Exception as e:
#                     logger.warning(f"Unexpected error parsing metadata: {str(e)} for value: '{x}'")
#                     return {}
            
#             df['metadata'] = df['metadata'].apply(parse_metadata)
            
#             logger.info(f"Successfully processed CSV with {len(df)} rows")
#             return df
            
#         except Exception as e:
#             logger.error(f"Error processing CSV: {str(e)}")
#             raise ValueError(f"Failed to process CSV file: {str(e)}")
    
#     def create_behavior_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
#         """Create a behavior matrix for clustering and return user IDs and the full user-action DataFrame."""
#         # Get unique actions
#         actions = df['action'].unique()
        
#         # Create user-action matrix
#         user_actions_df = pd.crosstab(df['user_id'], df['action'])
        
#         # Fill missing actions with 0
#         for action in actions:
#             if action not in user_actions_df.columns:
#                 user_actions_df[action] = 0
        
#         return user_actions_df.values, user_actions_df.index.tolist(), user_actions_df
    
#     def cluster_users(self, behavior_matrix: np.ndarray) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
#         """Cluster users using UMAP + HDBSCAN."""
#         # Reduce dimensionality with UMAP
#         reducer = umap.UMAP(n_components=2, random_state=42, spread=0.5, min_dist=0.001)
#         embeddings = reducer.fit_transform(behavior_matrix)
        
#         # Cluster with HDBSCAN
#         clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE)
#         cluster_labels = clusterer.fit_predict(embeddings)
        
#         return embeddings, clusterer
    
#     def get_cluster_feature_importance(self, user_actions_df: pd.DataFrame, cluster_id: int, df: pd.DataFrame) -> Dict:
#         """Identify top behavioral features differentiating a cluster."""
#         # Get user IDs for the current cluster
#         cluster_user_ids = df[df['cluster'] == cluster_id]['user_id'].unique()
        
#         # Filter the user_actions_df for this cluster
#         cluster_user_actions = user_actions_df.loc[cluster_user_ids]
        
#         # Calculate mean action frequency for the cluster
#         cluster_mean_actions = cluster_user_actions.mean()
        
#         # Calculate overall mean action frequency
#         overall_mean_actions = user_actions_df.mean()
        
#         # Calculate the ratio of cluster mean to overall mean (avoid division by zero)
#         # Add a small epsilon to avoid division by zero for overall_mean_actions that are 0
#         ratio = cluster_mean_actions / (overall_mean_actions + 1e-6)
        
#         # Identify top positive and negative differentiating features
#         # Positive: actions much more frequent in this cluster
#         # Negative: actions much less frequent in this cluster (compared to other actions)

#         # We'll use a threshold, e.g., ratio > 1.5 for more frequent, ratio < 0.5 for less frequent
#         differentiating_features = {
#             'highly_frequent': ratio[ratio > 1.5].sort_values(ascending=False).index.tolist(),
#             'less_frequent': ratio[ratio < 0.5].sort_values(ascending=True).index.tolist()
#         }

#         return differentiating_features

#     def get_cluster_summary(self, df: pd.DataFrame, cluster_id: int) -> Dict:
#         """Generate summary statistics for a cluster."""
#         cluster_users = df[df['cluster'] == cluster_id]['user_id'].unique()
#         cluster_data = df[df['user_id'].isin(cluster_users)]
        
#         # Initialize detailed summaries
#         feature_usage_details = {}
#         feedback_comments = []
#         proposal_status = {'created': 0, 'revised': 0, 'closed': 0, 'viewed_only': 0}
        
#         # Iterate through cluster data to gather detailed info
#         for _, row in cluster_data.iterrows():
#             action = row['action']
#             metadata = row['metadata']
            
#             if action == 'feature_use':
#                 feature_name = metadata.get('feature_name')
#                 if feature_name:
#                     feature_usage_details[feature_name] = feature_usage_details.get(feature_name, 0) + 1
#             elif action == 'feedback':
#                 comment = metadata.get('comment')
#                 if comment:
#                     feedback_comments.append(comment)
#             elif action == 'proposal_create':
#                 proposal_status['created'] += 1
#             elif action == 'proposal_revision':
#                 proposal_status['revised'] += 1
#             elif action == 'proposal_close':
#                 proposal_status['closed'] += 1
#             elif action == 'proposal_view' and action not in cluster_data[cluster_data['user_id'] == row['user_id']]['action'].values:
#                 # This is a simplification; a more robust check would involve checking for subsequent close actions
#                 proposal_status['viewed_only'] += 1

#         # Calculate basic stats
#         stats = {
#             'user_count': len(cluster_users),
#             'total_actions': len(cluster_data),
#             'avg_actions_per_user': len(cluster_data) / len(cluster_users),
#             'common_actions': cluster_data['action'].value_counts().head(5).to_dict(),
#             'time_range': {
#                 'start': cluster_data['timestamp'].min().isoformat(),
#                 'end': cluster_data['timestamp'].max().isoformat()
#             },
#             'feature_usage_details': feature_usage_details,
#             'feedback_comments': feedback_comments,
#             'proposal_flow_status': proposal_status,
#             'differentiating_features': self.get_cluster_feature_importance(self.user_actions_df, cluster_id, df)
#         }
        
#         # Calculate conversion and completion rates based on proposal flow
#         # Conversion Rate: Users who closed a proposal / Users who created a proposal
#         created_proposals = proposal_status.get('created', 0)
#         closed_proposals = proposal_status.get('closed', 0)
        
#         stats['conversion_rate'] = (closed_proposals / created_proposals) * 100 if created_proposals > 0 else 0
        
#         # Completion Rate: Users who closed a proposal / Total unique users in cluster (simplified)
#         # Or, users who closed a proposal / users who interacted with proposals (created, revised, viewed)
#         total_proposal_interactions = created_proposals + proposal_status.get('revised', 0) + proposal_status.get('viewed_only', 0)
#         stats['completion_rate'] = (closed_proposals / total_proposal_interactions) * 100 if total_proposal_interactions > 0 else 0

#         # Calculate funnel metrics: number of unique users at each stage
#         # For each stage, count unique users who performed that action
#         funnel_metrics = {
#             'product_view': len(cluster_data[cluster_data['action'] == 'product_view']['user_id'].unique()),
#             'search': len(cluster_data[cluster_data['action'] == 'search']['user_id'].unique()),
#             'filter': len(cluster_data[cluster_data['action'] == 'filter']['user_id'].unique()),
#             'add_to_cart': len(cluster_data[cluster_data['action'] == 'add_to_cart']['user_id'].unique()),
#             'remove_from_cart': len(cluster_data[cluster_data['action'] == 'remove_from_cart']['user_id'].unique()),
#             'abandon_cart': len(cluster_data[cluster_data['action'] == 'abandon_cart']['user_id'].unique()),
#             'checkout_start': len(cluster_data[cluster_data['action'] == 'checkout_start']['user_id'].unique()),
#             'purchase_complete': len(cluster_data[cluster_data['action'] == 'purchase_complete']['user_id'].unique())
#         }
#         stats['funnel_metrics'] = funnel_metrics

#         return stats
    
# def generate_markdown_report(cluster_summary: Dict, insights: Dict, funnel_df: pd.DataFrame) -> str:
#     """Generate a Markdown report with cluster insights and visualizations."""
#     # Create markdown content
#     md_content = f"""# Cluster Analysis Report

# ## Cluster Overview
# - **Name**: {insights['cluster_name']}
# - **Persona**: {insights['persona']}
# - **Metric Summary**: {insights['metric_summary_statement']}

# ## Key Performance Indicators
# - **Conversion Rate**: {cluster_summary['conversion_rate']:.2f}%
# - **Completion Rate**: {cluster_summary['completion_rate']:.2f}%

# ## User Journey Funnel
# | Stage | Users | Percentage |
# |-------|-------|------------|
# """
    
#     # Add funnel data
#     for _, row in funnel_df.iterrows():
#         md_content += f"| {row['Stage']} | {row['Users']} | {row['Percentage']:.2f}% |\n"
    
#     # Add pain points and hypotheses
#     md_content += "\n## Pain Points & Hypotheses\n"
#     for point, hypothesis in zip(insights['pain_points'], insights.get('hypotheses', [])):
#         md_content += f"### Pain Point: {point}\n"
#         md_content += f"**Hypothesis**: {hypothesis}\n\n"
    
#     # Add recommendations
#     md_content += "## Recommendations\n"
#     for i, rec in enumerate(insights['recommendations'], 1):
#         md_content += f"{i}. {rec}\n"
    
#     return md_content

# def simulate_ab_test(recommendation: str) -> Dict:
#     """Simulate setting up an A/B test for a recommendation."""
#     # Generate a mock A/B test configuration
#     test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
#     # Parse the recommendation to extract key elements
#     elements = recommendation.lower().split()
    
#     # Generate mock metrics
#     metrics = {
#         "primary_metric": "completion_rate",
#         "secondary_metrics": ["time_to_complete", "user_satisfaction"],
#         "expected_improvement": f"{np.random.randint(5, 25)}%",
#         "minimum_sample_size": np.random.randint(100, 1000),
#         "test_duration_days": np.random.randint(7, 30)
#     }
    
#     return {
#         "test_id": test_id,
#         "recommendation": recommendation,
#         "metrics": metrics,
#         "status": "configured",
#         "created_at": datetime.now().isoformat()
#     }

def main():
    st.set_page_config(
        page_title="Shopping Insights Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for modern styling
    st.markdown("""
        <style>
        .main {
            background-color: #1a1a1a; /* Dark background for the main content */
        }
        .stApp {
            max-width: unset; /* Use full width */
            margin: 0; /* Remove margin */
        }
        .title-text {
            font-size: 3rem;
            font-weight: 700;
            color: #ffffff; /* Light color for titles */
            margin-bottom: 1rem;
        }
        .subtitle-text {
            font-size: 1.5rem;
            color: #cccccc; /* Lighter color for subtitles */
            margin-bottom: 2rem;
        }
        .feature-card {
            background-color: #2b2b2b; /* Slightly lighter dark for cards */
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .feature-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #ffffff; /* Light color for feature titles */
            margin-bottom: 0.5rem;
        }
        .feature-text {
            color: #eeeeee; /* Light color for feature text */
            font-size: 1rem;
        }
        .code-block {
            background-color: #3a3a3a; /* Darker background for code blocks */
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
            color: #e0e0e0; /* Light color for code text */
        }
        .download-button {
            background-color: #6200EE; /* A distinct, dark-mode friendly color */
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            text-decoration: none;
            display: inline-block;
            margin: 1rem 0;
        }
        /* Adjust general text color for better readability in dark mode */
        p,
        li,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        label {
            color: #ffffff; /* Ensure all general text is white */
        }
        /* Adjust column text color */
        [data-testid="stVerticalBlock"] > div > div > div > div > div > p {
            color: #ffffff;
        }
        /* Streamlit's default components might need specific overrides */
        .stMarkdown, .stText, .stJson {
            color: #ffffff;
        }
        .stAlert {
            background-color: #3a3a3a;
            color: #ffffff;
        }
        .stExpander div[data-testid="stExpanderToggle"] > p {
            color: #ffffff;
        }
        .stTextInput label > div > p,
        .stTextArea label > div > p,
        .stSelectbox label > div > p,
        .stRadio label > div > p {
            color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'roadmap' not in st.session_state:
        st.session_state.roadmap = []
    if 'ab_tests' not in st.session_state:
        st.session_state.ab_tests = []
    
    # Initialize session state for insights and cluster summary
    if 'current_insights' not in st.session_state:
        st.session_state.current_insights = None
    if 'current_cluster_summary' not in st.session_state:
        st.session_state.current_cluster_summary = None

    # Sidebar
    st.sidebar.title("Shopping Insights")
    
    # File upload in sidebar
    uploaded_file = st.sidebar.file_uploader("Upload User Behavior CSV", type=['csv'])
    
    if uploaded_file is None:
        # Modern, clean welcome
        st.markdown('<div class="title-text">🛒 Shopping Website Behavioral Insights Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle-text">Upload your user interaction data and let AI uncover hidden pain points, drop-offs, and opportunities in your shopping journey.</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown('## 🚀 What You Can Do Here')
            st.markdown("""
            - **Segment Users**: Use clustering to identify behavior-based groups.
            - **Diagnose Drop-Offs**: Pinpoint where users exit the shopping journey.
            - **Get Smart Insights**: Leverage AI for actionable recommendations.
            - **Plan Experiments**: Simulate A/B tests and generate hypotheses.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

            # Feature grid
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.markdown('### 📊 User Segmentation')
                st.markdown("- UMAP + HDBSCAN visualizations\n- Interactive, labeled plots")
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.markdown('### 🛍️ Shopping Journey Mapping')
                st.markdown("- Funnel breakdown\n- Conversion and completion metrics")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.markdown('### 🤖 AI Recommendations')
                st.markdown("- PM, tech, and UX tones\n- JSON-based insights + hypotheses")
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.markdown('### 🧪 Test Simulation')
                st.markdown("- One-click A/B test simulation\n- Estimated lift, duration, sample size")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown('## 📝 Upload Format Requirements')
            st.markdown("""
            Your CSV file should have the following columns:

            - `user_id`: Unique identifier for each user  
            - `timestamp`: ISO format timestamps (e.g., 2024-03-20T10:00:00)  
            - `action`: The type of user interaction (e.g., `product_view`, `add_to_cart`, `purchase_complete`)  
            - `metadata`: JSON metadata related to the action
            """)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown('### 📁 Sample Data Preview')
            
            # Read the actual synthetic_demo_100.csv file
            try:
                with open('synthetic_demo_100.csv', 'r') as file:
                    sample_data = file.read()
                st.code(sample_data, language="csv")
            except FileNotFoundError:
                # Fallback to hardcoded sample if file not found
                st.code("""user_id,timestamp,action,metadata
user1,2024-03-20T10:00:00,product_view,{"product_id":"A1"}
user1,2024-03-20T10:01:00,search,{"query":"shoes"}
user1,2024-03-20T10:02:00,filter,{"filter":"size:9"}
user1,2024-03-20T10:03:00,add_to_cart,{"product_id":"A1"}
user1,2024-03-20T10:04:00,checkout_start,{"cart_value":100}
user1,2024-03-20T10:05:00,purchase_complete,{"order_id":"O1001"}
user2,2024-03-20T11:00:00,product_view,{"product_id":"B2"}
user2,2024-03-20T11:01:00,add_to_cart,{"product_id":"B2"}
user2,2024-03-20T11:02:00,remove_from_cart,{"product_id":"B2"}
user2,2024-03-20T11:03:00,abandon_cart,{}
user3,2024-03-20T12:00:00,product_view,{"product_id":"C3"}
user3,2024-03-20T12:01:00,add_to_cart,{"product_id":"C3"}
user3,2024-03-20T12:02:00,checkout_start,{"cart_value":50}
user3,2024-03-20T12:03:00,purchase_complete,{"order_id":"O1002"}
""", language="csv")

            # Download button for the actual synthetic_demo_100.csv file
            try:
                with open('synthetic_demo_100.csv', 'r') as file:
                    csv_data = file.read()
                st.download_button(
                    label="⬇ Download Sample CSV",
                    data=csv_data,
                    file_name="synthetic_demo_100.csv",
                    mime="text/csv"
                )
            except FileNotFoundError:
                # Fallback to hardcoded sample if file not found
                st.download_button(
                    label="⬇ Download Sample CSV",
                    data="""user_id,timestamp,action,metadata
user1,2024-03-20T10:00:00,product_view,{"product_id":"A1"}
user1,2024-03-20T10:01:00,search,{"query":"shoes"}
user1,2024-03-20T10:02:00,filter,{"filter":"size:9"}
user1,2024-03-20T10:03:00,add_to_cart,{"product_id":"A1"}
user1,2024-03-20T10:04:00,checkout_start,{"cart_value":100}
user1,2024-03-20T10:05:00,purchase_complete,{"order_id":"O1001"}
user2,2024-03-20T11:00:00,product_view,{"product_id":"B2"}
user2,2024-03-20T11:01:00,add_to_cart,{"product_id":"B2"}
user2,2024-03-20T11:02:00,remove_from_cart,{"product_id":"B2"}
user2,2024-03-20T11:03:00,abandon_cart,{}
user3,2024-03-20T12:00:00,product_view,{"product_id":"C3"}
user3,2024-03-20T12:01:00,add_to_cart,{"product_id":"C3"}
user3,2024-03-20T12:02:00,checkout_start,{"cart_value":50}
user3,2024-03-20T12:03:00,purchase_complete,{"order_id":"O1002"}
""",
                    file_name="sample_behavior_data.csv",
                    mime="text/csv"
                )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""---
    👉 Use the file uploader in the sidebar to get started!
    """)

    
    else:
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable.")
            return

        # Initialize analyzer
        # analyzer = BehavioralAnalyzer(api_key)
        llm_agent = LLMAgent(api_key)
        clustering_analyzer = ClusteringAnalyzer()
        
        try:
            # Process CSV
            df = process_csv(uploaded_file)
            
            # Create behavior matrix
            behavior_matrix, user_ids_for_clustering, user_actions_df = clustering_analyzer.create_behavior_matrix(df)
            clustering_analyzer.user_actions_df = user_actions_df
            
            # Cluster users
            embeddings, clusterer = clustering_analyzer.cluster_users(behavior_matrix)
            
            # Create a DataFrame for user_id and cluster labels from the clustering input
            user_to_cluster = pd.DataFrame({
                'user_id': user_ids_for_clustering,
                'cluster': clusterer.labels_,
                'cluster_label': [f'Cluster {x}' if x != -1 else 'Noise' for x in clusterer.labels_]
            })
            
            # Merge cluster labels back to the original DataFrame
            df = df.merge(user_to_cluster, on='user_id', how='left')
            
            # Create scatter plot
            plot_df = pd.DataFrame({
                'user_id': user_ids_for_clustering,
                'x': embeddings[:, 0],
                'y': embeddings[:, 1],
                'cluster': clusterer.labels_,
                'cluster_label': user_to_cluster['cluster_label']
            })

            # Filter out 'Noise' cluster from the plot_df
            plot_df = plot_df[plot_df['cluster_label'] != 'Noise']

            # Calculate cluster sizes and merge with plot_df
            cluster_sizes = plot_df.groupby('cluster_label').size().reset_index(name='cluster_size')
            plot_df = plot_df.merge(cluster_sizes, on='cluster_label', how='left')

            # Main content area
            st.title("Behavioral Analysis Dashboard")
            
            # Display basic stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Users", len(df['user_id'].unique()))
            with col2:
                st.metric("Total Actions", len(df))
            with col3:
                st.metric("Unique Clusters", len(plot_df['cluster_label'].unique()) - 1)  # Exclude Noise

            # Display plot
            st.subheader("User Clusters")
            fig = px.scatter(
                plot_df,
                x='x',
                y='y',
                color='cluster_label',
                size='cluster_size', # Scale size by cluster_size
                size_max=10,       # Max size for largest bubbles
                hover_data={'user_id': True, 'cluster': True, 'cluster_label': True, 'cluster_size': True},
                title="User Clusters"
            )
            
            # Update axis labels
            fig.update_layout(
                xaxis_title="Behavioral Dimension 1",
                yaxis_title="Behavioral Dimension 2"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Handle cluster selection
            selected_cluster_label = st.selectbox(
                "Select Cluster",
                options=sorted(plot_df['cluster_label'].unique()),
                format_func=lambda x: x,
                key="cluster_selectbox"
            )
            
            # Map the selected label back to the numerical cluster ID for filtering
            selected_cluster_id = plot_df[plot_df['cluster_label'] == selected_cluster_label]['cluster'].iloc[0]

            if selected_cluster_label != 'Noise':
                # Get cluster summary
                cluster_summary = clustering_analyzer.get_cluster_summary(df, selected_cluster_id)
                st.session_state.current_cluster_summary = cluster_summary # Store in session state
                
                # Add controls for output mode and tone
                col1, col2 = st.columns(2)
                with col1:
                    output_mode = st.radio(
                        "Output Mode",
                        options=["pm", "technical"],
                        format_func=lambda x: "Product Manager" if x == "pm" else "Technical",
                        horizontal=True,
                        key="output_mode_radio"
                    )
                with col2:
                    tone = st.selectbox(
                        "Analysis Tone",
                        options=["default", "ux_designer", "developer", "business"],
                        format_func=lambda x: x.replace("_", " ").title(),
                        key="analysis_tone_selectbox"
                    )
                
                # Get LLM insights with selected mode and tone and store in session state
                insights = llm_agent.get_llm_insights(cluster_summary, output_mode=output_mode, tone=tone)
                st.session_state.current_insights = insights # Store in session state
                
                # Display cluster information - retrieve from session state
                insights_to_display = st.session_state.current_insights
                cluster_summary_to_display = st.session_state.current_cluster_summary

                st.subheader(f"Cluster: {insights_to_display['cluster_name']}")
                st.markdown(f"**Persona**: {insights_to_display['persona']}")
                st.markdown(f"**Metric Summary**: {insights_to_display['metric_summary_statement']}")
                
                # Display KPIs
                st.subheader("Key Performance Indicators (KPIs)")
                st.markdown(f"**Conversion Rate (Purchase Complete/Checkout Start)**: {cluster_summary_to_display['conversion_rate']:.2f}%")
                st.markdown(f"**Completion Rate (Purchase Complete/Add to Cart)**: {cluster_summary_to_display['completion_rate']:.2f}%")

                # Display funnel visualization
                st.subheader("User Journey Funnel")
                funnel_metrics = cluster_summary_to_display['funnel_metrics']
                funnel_data = {
                    'Stage': [
                        'Product View', 'Search', 'Filter', 'Add to Cart', 'Remove from Cart',
                        'Abandon Cart', 'Checkout Start', 'Purchase Complete'
                    ],
                    'Users': [
                        funnel_metrics['product_view'],
                        funnel_metrics['search'],
                        funnel_metrics['filter'],
                        funnel_metrics['add_to_cart'],
                        funnel_metrics['remove_from_cart'],
                        funnel_metrics['abandon_cart'],
                        funnel_metrics['checkout_start'],
                        funnel_metrics['purchase_complete']
                    ]
                }
                funnel_df = pd.DataFrame(funnel_data)
                funnel_df['Percentage'] = (funnel_df['Users'] / funnel_df['Users'].iloc[0]) * 100 if funnel_df['Users'].iloc[0] > 0 else 0
                fig_funnel = px.funnel(funnel_df, x='Percentage', y='Stage', title=f'User Journey Funnel - {selected_cluster_label}')
                st.plotly_chart(fig_funnel, use_container_width=True)

                # Display pain points and hypotheses - retrieve from session state
                st.subheader("Pain Points & Testable Hypotheses")
                for i, (point, hypothesis) in enumerate(zip(insights_to_display['pain_points'], insights_to_display.get('hypotheses', []))):
                    with st.expander(f"Pain Point {i+1}: {point}"):
                        st.markdown(f"**Hypothesis**: {hypothesis}")
                
                # Display recommendations with A/B test simulation - retrieve from session state
                st.subheader("Recommendations")
                for i, rec in enumerate(insights_to_display['recommendations']):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"{i+1}. **{rec}**")
                    with col2:
                        if st.button(f"Simulate A/B Test", key=f"ab_test_{i}"):
                            test_config = simulate_ab_test(rec)
                            st.session_state.ab_tests.append(test_config)
                            st.success(f"A/B test configured for: {rec}")
                            # Display the just-simulated test in a new expander immediately
                            with st.expander(f"Simulated Test Details: {test_config['test_id']}"):
                                st.json(test_config)

                # Export functionality - retrieve from session state
                st.subheader("Export Report")
                if st.button("Generate Report"):
                    markdown_content = generate_markdown_report(cluster_summary_to_display, insights_to_display, funnel_df)
                    st.download_button(
                        label="Download Markdown Report",
                        data=markdown_content,
                        file_name=f"cluster_analysis_{selected_cluster_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                    
                    # Also display the report in the app
                    with st.expander("Preview Report"):
                        st.markdown(markdown_content)

                # Display active A/B tests
                if st.session_state.ab_tests:
                    st.subheader("Active A/B Tests")
                    for test in st.session_state.ab_tests:
                        with st.expander(f"Test {test['test_id']}"):
                            st.json(test)

                with st.expander("View Raw Cluster Summary"):
                    st.json(cluster_summary_to_display)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 