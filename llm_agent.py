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
        # Use OpenRouter base URL
        self.openai_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        
    def get_llm_insights(self, cluster_summary: Dict, output_mode: str = "pm", tone: str = "default") -> Dict:
        """Get pain points and recommendations from LLM with different output modes and tones, optimized to focus on drop-offs and conversion bottlenecks."""

        # Construct detailed statistics string
        detailed_stats = f"""
        Detailed Metrics:
        - Specific Feature Usage: {cluster_summary['feature_usage_details']}
        - User Feedback Comments: {cluster_summary['feedback_comments']}
        - Proposal Flow Status (Created/Revised/Closed/Viewed Only): {cluster_summary['proposal_flow_status']}
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
            "pm": """You are analyzing a user segment on a proposal-based platform (like Grupa.io). Based on the metrics and behavioral patterns, your job is to:
            1. Identify the **core behavioral intent** of users in this segment.
            2. Determine **where in the proposal flow they are dropping off**, even if they interact with later stages (e.g., viewing without closing).
            3. Focus on **conversion bottlenecks** — not just frequency of usage, but where users abandon or fail to complete expected flows.
            4. Provide:

                - **Cluster Name**: A concise name (e.g., "Stuck Reviewers", "Abandoned Creators").
                - **Persona**: A 1-2 line summary of user behavior and goal.
                - **Metric Summary**: Key behavioral insights, including drop-off ratios.
                - **Pain Points**: Focused on product flow gaps (e.g., friction between proposal view and close).
                - **Product Recommendations**: Strategic changes to help users fulfill intent.
                - **Hypotheses**: For each pain point, give a testable hypothesis in the format: 
                "[specific change] will [expected outcome] by [quantified improvement]."
            """,
                    "technical": """You are analyzing user segments for a product engineering team. Prioritize:
            1. Identifying user flow breakdowns in the proposal lifecycle.
            2. Highlighting points of UI/UX friction that prevent conversion.
            3. Recommending backend/frontend changes to support smoother transitions (e.g., saving drafts, pre-fill content, nudges).

            Provide:
            - Cluster technical name
            - Implementation-relevant persona
            - Metric summary focused on engagement delta across funnel steps
            - Technical pain points and recommendations
            - Hypotheses for A/B testing improvements"""
                }

        # Final prompt string
        prompt = f"""You are a product analytics assistant analyzing a user cluster.
        Please infer and explain where users drop off in their journey — for example, Who drops off after visiting but doesn't start a proposal?, Who creates proposals but never submits them?, Who are your power users? etc.
        Do not just describe the most frequent actions — focus on user intent and where it breaks.

        Cluster Overview:
        - Number of users: {cluster_summary['user_count']}
        - Total actions: {cluster_summary['total_actions']}
        - Average actions per user: {cluster_summary['avg_actions_per_user']}
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
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            insights = json.loads(response.choices[0].message.content)
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