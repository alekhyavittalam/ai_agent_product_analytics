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
                
                # Define the complete funnel stages in order
                all_stages = [
                    'product_view', 'search', 'filter', 'add_to_cart', 
                    'remove_from_cart', 'abandon_cart', 'checkout_start', 'purchase_complete'
                ]
                
                # Create DataFrame with all stages, filling in zeros for missing ones
                funnel_data = []
                for stage in all_stages:
                    users = funnel_metrics.get(stage, 0)
                    funnel_data.append({
                        'Stage': stage.replace('_', ' ').title(),
                        'Users': users
                    })
                
                funnel_df = pd.DataFrame(funnel_data)
                
                # Create the funnel chart
                fig_funnel = px.funnel(
                    funnel_df, 
                    x='Users', 
                    y='Stage', 
                    title=f'User Journey Funnel - {selected_cluster_label}'
                )
                
                # Customize the chart to show zero values clearly
                fig_funnel.update_layout(
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                )
                
                st.plotly_chart(fig_funnel, use_container_width=True)
                
                # Add a summary of the funnel
                total_users = funnel_df['Users'].max() if not funnel_df.empty else 0
                if total_users > 0:
                    st.markdown(f"**Total users in this cluster: {total_users}**")
                    
                    # Show conversion rates between key stages
                    product_view = funnel_df[funnel_df['Stage'] == 'Product View']['Users'].iloc[0] if len(funnel_df[funnel_df['Stage'] == 'Product View']) > 0 else 0
                    add_to_cart = funnel_df[funnel_df['Stage'] == 'Add To Cart']['Users'].iloc[0] if len(funnel_df[funnel_df['Stage'] == 'Add To Cart']) > 0 else 0
                    checkout_start = funnel_df[funnel_df['Stage'] == 'Checkout Start']['Users'].iloc[0] if len(funnel_df[funnel_df['Stage'] == 'Checkout Start']) > 0 else 0
                    purchase_complete = funnel_df[funnel_df['Stage'] == 'Purchase Complete']['Users'].iloc[0] if len(funnel_df[funnel_df['Stage'] == 'Purchase Complete']) > 0 else 0
                    
                    if product_view > 0:
                        browse_to_cart_rate = (add_to_cart / product_view) * 100
                        st.markdown(f"**Browse to Cart Rate**: {browse_to_cart_rate:.1f}%")
                    
                    if add_to_cart > 0:
                        cart_to_checkout_rate = (checkout_start / add_to_cart) * 100
                        st.markdown(f"**Cart to Checkout Rate**: {cart_to_checkout_rate:.1f}%")
                    
                    if checkout_start > 0:
                        checkout_to_purchase_rate = (purchase_complete / checkout_start) * 100
                        st.markdown(f"**Checkout to Purchase Rate**: {checkout_to_purchase_rate:.1f}%")

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