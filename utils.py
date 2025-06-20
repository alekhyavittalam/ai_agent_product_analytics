import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserBehavior:
    user_id: str
    timestamp: datetime
    action: str
    metadata: Dict

def process_csv(csv_path: str) -> pd.DataFrame:
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

def generate_markdown_report(cluster_summary: Dict, insights: Dict, funnel_df: pd.DataFrame) -> str:
    """Generate a Markdown report with cluster insights and visualizations."""
    # Create markdown content
    md_content = f"""# Cluster Analysis Report

## Cluster Overview
- **Name**: {insights['cluster_name']}
- **Persona**: {insights['persona']}
- **Metric Summary**: {insights['metric_summary_statement']}

## Key Performance Indicators
- **Conversion Rate**: {cluster_summary['conversion_rate']:.2f}%
- **Completion Rate**: {cluster_summary['completion_rate']:.2f}%

## User Journey Funnel
| Stage | Users |
|-------|-------|
"""
    
    # Add funnel data - use the new structure
    for _, row in funnel_df.iterrows():
        md_content += f"| {row['Stage']} | {row['Users']} |\n"
    
    # Add pain points and hypotheses
    md_content += "\n## Pain Points & Hypotheses\n"
    for point, hypothesis in zip(insights['pain_points'], insights.get('hypotheses', [])):
        md_content += f"### Pain Point: {point}\n"
        md_content += f"**Hypothesis**: {hypothesis}\n\n"
    
    # Add recommendations
    md_content += "## Recommendations\n"
    for i, rec in enumerate(insights['recommendations'], 1):
        md_content += f"{i}. {rec}\n"
    
    return md_content

def simulate_ab_test(recommendation: str) -> Dict:
    """Simulate setting up an A/B test for a recommendation."""
    # Generate a mock A/B test configuration
    test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Parse the recommendation to extract key elements
    elements = recommendation.lower().split()
    
    # Generate mock metrics
    metrics = {
        "primary_metric": "completion_rate",
        "secondary_metrics": ["time_to_complete", "user_satisfaction"],
        "expected_improvement": f"{np.random.randint(5, 25)}%",
        "minimum_sample_size": np.random.randint(100, 1000),
        "test_duration_days": np.random.randint(7, 30)
    }
    
    return {
        "test_id": test_id,
        "recommendation": recommendation,
        "metrics": metrics,
        "status": "configured",
        "created_at": datetime.now().isoformat()
    } 