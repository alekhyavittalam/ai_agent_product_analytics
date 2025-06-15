# Grupa.io Behavioral Analysis

A Streamlit application for analyzing user behavior patterns and generating product recommendations.

## Features

- Upload and process user behavior CSV files
- Cluster users based on behavioral patterns using UMAP + HDBSCAN
- Interactive visualization of user clusters
- AI-powered analysis of pain points and recommendations
- A/B test roadmap management

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

3. Run the application:
```bash
streamlit run app.py
```

## CSV Format

The application expects a CSV file with the following columns:
- `user_id`: Unique identifier for each user
- `timestamp`: When the action occurred (ISO format)
- `action`: The type of action performed
- `metadata`: JSON string containing additional action details

Example:
```csv
user_id,timestamp,action,metadata
user1,2024-03-20T10:00:00,proposal_create,{"proposal_id": "123", "type": "investment"}
user1,2024-03-20T10:30:00,proposal_revision,{"proposal_id": "123", "revision": 1}
```

## Usage

1. Upload your CSV file using the sidebar
2. View the interactive cluster visualization
3. Select a cluster to see detailed analysis
4. Add recommendations to your A/B test roadmap
5. Manage your roadmap in the sidebar

## Development

The application is built with:
- Streamlit for the UI
- UMAP + HDBSCAN for clustering
- Plotly for visualizations
- OpenAI GPT-4 for insights generation