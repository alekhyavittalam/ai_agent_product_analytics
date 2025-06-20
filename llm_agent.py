import os
import json
import logging
from openai import OpenAI
from typing import Dict

# Configure logging
logger = logging.getLogger(__name__)

class LLMAgent:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenAI API Key not provided or found in environment variables.")
        # Use OpenRouter with your API key
        self.openai_client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
    def get_llm_insights(self, cluster_summary: Dict, output_mode: str = "pm", tone: str = "default") -> Dict:
        """Get pain points and recommendations from LLM with different output modes and tones, optimized to focus on drop-offs and conversion bottlenecks."""

        # Construct detailed statistics string
        detailed_stats = f"""
        Detailed Metrics:
        - Specific Feature Usage: {cluster_summary['feature_usage_details']}
        - User Feedback Comments: {cluster_summary['feedback_comments']}
        - Shopping Flow Status (Product View/Search/Filter/Add to Cart/Remove from Cart/Abandon Cart/Checkout Start/Purchase Complete): {cluster_summary['shopping_flow_status']}
        - Differentiating Behavioral Features: {cluster_summary['differentiating_features']}
        """

        # Define tone-specific instructions
        tone_instructions = {
            "default": "Provide clear, professional analysis.",
            "ux_designer": "Focus on user experience, interaction patterns, and design implications.",
            "developer": "Emphasize technical implementation details and code-level considerations.",
            "business": "Highlight business impact, ROI, and strategic implications."
        }

        # Define output mode-specific instructions
        mode_instructions = {
            "pm": """You are analyzing a user segment on a shopping website. Based on the metrics and behavioral patterns, your job is to:
            1. Identify the **core behavioral intent** of users in this segment.
            2. Determine **where in the shopping journey they are dropping off**, even if they interact with later stages (e.g., add to cart without purchasing).
            3. Focus on **conversion bottlenecks** — not just frequency of usage, but where users abandon or fail to complete expected flows.
            4. Provide:

                - **Cluster Name**: A concise name (e.g., 'Cart Abandoners', 'Window Shoppers').
                - **Persona**: A 1-2 line summary of user behavior and goal.
                - **Metric Summary**: Key behavioral insights, including drop-off ratios.
                - **Pain Points**: Focused on product flow gaps (e.g., friction between add to cart and checkout).
                - **Product Recommendations**: Strategic changes to help users fulfill intent.
                - **Hypotheses**: For each pain point, give a testable hypothesis in the format: 
                '[specific change] will [expected outcome] by [quantified improvement].'
            """,
            "technical": """You are analyzing user segments for a shopping website product engineering team. Prioritize:
            1. Identifying user flow breakdowns in the shopping journey (product view, search, filter, add to cart, remove from cart, abandon cart, checkout start, purchase complete).
            2. Highlighting points of UI/UX friction that prevent conversion.
            3. Recommending backend/frontend changes to support smoother transitions (e.g., persistent carts, better search/filter UX, checkout nudges).

            Provide:
            - Cluster technical name
            - Implementation-relevant persona
            - Metric summary focused on engagement delta across funnel steps
            - Technical pain points and recommendations
            - Hypotheses for A/B testing improvements"""
        }

        # Final prompt string
        prompt = f"""You are a product analytics assistant analyzing a user cluster.
        Your primary goal is to create a specific and descriptive persona for this cluster.

        **Crucially, you must incorporate a time-based analysis into your response.** Analyze the 'time_range' provided.
        - If the duration between the start and end time is short (e.g., minutes or hours), it implies users are acting quickly.
        - If the duration is long (e.g., multiple days), it implies a slower, more deliberate user journey.
        Use this temporal insight to make the cluster name more specific. For example, instead of just 'Cart Abandoners', you could have 'Quick Cart Abandoners' for users who abandon within minutes, or 'Multi-Day Cart Abandoners' for those who take longer.

        Please infer and explain where users drop off in their shopping journey. Do not just describe the most frequent actions — focus on user intent and where it breaks.

        Cluster Overview:
        - Number of users: {cluster_summary['user_count']}
        - Total actions: {cluster_summary['total_actions']}
        - Average actions per user: {cluster_summary['avg_actions_per_user']:.2f}
        - Most common actions: {cluster_summary['common_actions']}
        - Time range: {cluster_summary['time_range']['start']} to {cluster_summary['time_range']['end']}
        {detailed_stats}

        {tone_instructions.get(tone, tone_instructions['default'])}
        {mode_instructions.get(output_mode, mode_instructions['pm'])}

        Output as valid JSON with these keys:
        - 'cluster_name'
        - 'persona'
        - 'metric_summary_statement'
        - 'pain_points' (list of 3)
        - 'recommendations' (list of 3)
        - 'hypotheses' (1 per pain point)
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            # Parse the response into JSON
            try:
                insights = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response
                content = response.choices[0].message.content
                insights = {
                    'cluster_name': "Analysis Results",
                    'persona': content[:200],  # First 200 characters as persona
                    'metric_summary_statement': "See detailed analysis below",
                    'pain_points': [content],
                    'recommendations': ["Please review the analysis and formulate specific recommendations"],
                    'hypotheses': ["Based on the analysis, specific hypotheses can be formulated"]
                }
            
            return insights

        except Exception as e:
            logger.error(f"Error getting LLM insights: {str(e)}")
            return {
                'cluster_name': "Error getting insights",
                'persona': "Error getting insights",
                'metric_summary_statement': "Error getting insights",
                'pain_points': ["Error getting insights"],
                'recommendations': ["Error getting recommendations"],
                'hypotheses': ["Error generating hypotheses"]
            } 