import os
import json
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

@dataclass
class UserBehavior:
    user_id: str
    timestamp: datetime
    action: str
    metadata: Dict

class BehavioralSegmentationAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze_behavior(self, user_behaviors: List[UserBehavior]) -> Dict:
        """
        Analyze user behaviors and return segmentation results
        
        Args:
            user_behaviors: List of UserBehavior objects containing user actions and metadata
            
        Returns:
            Dict containing analysis results including:
            - user_patterns: Dictionary of user_id to their behavior patterns
            - action_frequencies: Frequency of different actions
            - time_based_patterns: Patterns based on time of actions
            - metadata_insights: Insights from metadata analysis
            - ux_insights: Insights about user experience and pain points
        """
        # Group behaviors by user
        user_behaviors_dict = {}
        for behavior in user_behaviors:
            if behavior.user_id not in user_behaviors_dict:
                user_behaviors_dict[behavior.user_id] = []
            user_behaviors_dict[behavior.user_id].append(behavior)
        
        # Initialize results
        user_patterns = {}
        action_frequencies = {}
        time_based_patterns = {}
        metadata_insights = {}
        ux_insights = {}
        
        # First, collect all basic statistics without API calls
        for user_id, behaviors in user_behaviors_dict.items():
            # Update action frequencies
            for behavior in behaviors:
                action_frequencies[behavior.action] = action_frequencies.get(behavior.action, 0) + 1
            
            # Analyze time-based patterns
            time_patterns = self._analyze_time_patterns(behaviors)
            time_based_patterns[user_id] = time_patterns
            
            # Analyze metadata
            metadata_insights[user_id] = self._analyze_metadata(behaviors)
            
            # Analyze UX patterns
            ux_insights[user_id] = self._analyze_ux_patterns(behaviors)
            
            # Prepare simplified data for API analysis
            behavior_summary = {
                "user_id": user_id,
                "total_actions": len(behaviors),
                "action_types": list(set(b.action for b in behaviors)),
                "time_range": {
                    "start": min(b.timestamp for b in behaviors).isoformat(),
                    "end": max(b.timestamp for b in behaviors).isoformat()
                }
            }
            
            # Call OpenRouter API for behavior analysis with reduced token usage
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": "anthropic/claude-3-opus-20240229",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Analyze this user's product usage patterns, focusing on their interaction style (e.g., fast closer, high reviser, ghost) and potential UX pain points. Provide key insights in 2-3 sentences."
                            },
                            {
                                "role": "user",
                                "content": json.dumps(behavior_summary)
                            }
                        ],
                        "max_tokens": 150
                    }
                )
                
                if response.status_code == 200:
                    analysis = response.json()
                    user_patterns[user_id] = analysis
                else:
                    error_msg = response.json().get("error", {}).get("message", "Unknown error")
                    print(f"Error analyzing behaviors for user {user_id}: {error_msg}")
                    user_patterns[user_id] = self._generate_basic_pattern_analysis(behaviors)
                    
            except Exception as e:
                print(f"Error processing user {user_id}: {str(e)}")
                user_patterns[user_id] = self._generate_basic_pattern_analysis(behaviors)
        
        return {
            "user_patterns": user_patterns,
            "action_frequencies": action_frequencies,
            "time_based_patterns": time_based_patterns,
            "metadata_insights": metadata_insights,
            "ux_insights": ux_insights
        }
    
    def _generate_basic_pattern_analysis(self, behaviors: List[UserBehavior]) -> Dict:
        """Generate basic pattern analysis without API call"""
        actions = [b.action for b in behaviors]
        unique_actions = set(actions)
        action_counts = {action: actions.count(action) for action in unique_actions}
        
        return {
            "basic_analysis": {
                "total_actions": len(behaviors),
                "unique_actions": list(unique_actions),
                "action_distribution": action_counts,
                "primary_action": max(action_counts.items(), key=lambda x: x[1])[0]
            }
        }
    
    def _analyze_time_patterns(self, behaviors: List[UserBehavior]) -> Dict:
        """Analyze time-based patterns in user behaviors"""
        patterns = {
            "time_of_day": {},
            "day_of_week": {},
            "frequency": len(behaviors)
        }
        
        for behavior in behaviors:
            # Analyze time of day
            hour = behavior.timestamp.hour
            time_of_day = "morning" if 5 <= hour < 12 else "afternoon" if 12 <= hour < 17 else "evening" if 17 <= hour < 22 else "night"
            patterns["time_of_day"][time_of_day] = patterns["time_of_day"].get(time_of_day, 0) + 1
            
            # Analyze day of week
            day = behavior.timestamp.strftime("%A")
            patterns["day_of_week"][day] = patterns["day_of_week"].get(day, 0) + 1
        
        return patterns
    
    def _analyze_metadata(self, behaviors: List[UserBehavior]) -> Dict:
        """Analyze metadata patterns in user behaviors"""
        insights = {
            "common_actions": {},
            "metadata_patterns": {}
        }
        
        for behavior in behaviors:
            # Track common actions
            insights["common_actions"][behavior.action] = insights["common_actions"].get(behavior.action, 0) + 1
            
            # Analyze metadata patterns
            for key, value in behavior.metadata.items():
                if key not in insights["metadata_patterns"]:
                    insights["metadata_patterns"][key] = {}
                if isinstance(value, (int, float)):
                    if "numeric_values" not in insights["metadata_patterns"][key]:
                        insights["metadata_patterns"][key]["numeric_values"] = []
                    insights["metadata_patterns"][key]["numeric_values"].append(value)
                else:
                    if "categorical_values" not in insights["metadata_patterns"][key]:
                        insights["metadata_patterns"][key]["categorical_values"] = {}
                    insights["metadata_patterns"][key]["categorical_values"][str(value)] = \
                        insights["metadata_patterns"][key]["categorical_values"].get(str(value), 0) + 1
        
        return insights
    
    def _analyze_ux_patterns(self, behaviors: List[UserBehavior]) -> Dict:
        """Analyze UX patterns and potential pain points"""
        pain_points = defaultdict(int)
        interaction_styles = defaultdict(int)
        drop_off_points = defaultdict(int)
        
        # Group behaviors by user
        user_behaviors = defaultdict(list)
        for behavior in behaviors:
            user_behaviors[behavior.user_id].append(behavior)
        
        for user_id, user_actions in user_behaviors.items():
            # Sort actions by timestamp
            user_actions.sort(key=lambda x: x.timestamp)
            
            # Track proposal states
            proposal_states = defaultdict(lambda: {
                'created': False,
                'revisions': 0,
                'last_action': None,
                'time_to_close': None,
                'drop_off': False,
                'feature_usage': set(),
                'session_duration': 0,
                'device_type': None
            })
            
            current_session_start = None
            current_session_duration = 0
            
            for action in user_actions:
                # Track session patterns
                if action.action == "session_start":
                    if current_session_start:
                        # Previous session ended
                        if current_session_duration < 300:  # Less than 5 minutes
                            pain_points["short_session_duration"] += 1
                    current_session_start = action.timestamp
                    current_session_duration = 0
                    if 'device' in action.metadata:
                        proposal_states[user_id]['device_type'] = action.metadata['device']
                elif current_session_start:
                    current_session_duration += 5  # Assume 5 minutes between actions
                
                # Track feature usage
                if action.action == "feature_use":
                    if 'feature_name' in action.metadata:
                        proposal_states[user_id]['feature_usage'].add(action.metadata['feature_name'])
                        if action.metadata.get('duration', 0) < 30:  # Less than 30 seconds
                            pain_points["quick_feature_abandonment"] += 1
                
                # Track proposal lifecycle
                if action.action == "proposal_create":
                    proposal_states[user_id]['created'] = True
                    proposal_states[user_id]['last_action'] = action.timestamp
                elif action.action == "proposal_revision":
                    proposal_states[user_id]['revisions'] += 1
                    proposal_states[user_id]['last_action'] = action.timestamp
                    if proposal_states[user_id]['revisions'] > 3:
                        pain_points["excessive_revisions"] += 1
                elif action.action == "proposal_close":
                    if proposal_states[user_id]['created']:
                        time_to_close = (action.timestamp - proposal_states[user_id]['last_action']).total_seconds() / 3600
                        proposal_states[user_id]['time_to_close'] = time_to_close
                        if time_to_close > 24:  # More than 24 hours
                            pain_points["slow_proposal_closure"] += 1
                elif action.action == "proposal_view":
                    if proposal_states[user_id]['created']:
                        view_duration = action.metadata.get('view_duration', 0)
                        if view_duration < 60:  # Less than 1 minute
                            pain_points["quick_proposal_views"] += 1
                        elif view_duration > 600:  # More than 10 minutes
                            pain_points["extended_proposal_views"] += 1
                
                # Track feedback patterns
                if action.action == "feedback":
                    if action.metadata.get('rating', 0) <= 2:
                        pain_points["low_ratings"] += 1
                    if 'comment' in action.metadata and len(action.metadata['comment']) < 10:
                        pain_points["minimal_feedback"] += 1
            
            # Analyze session patterns
            if current_session_duration > 3600:  # More than 1 hour
                pain_points["extended_sessions"] += 1
            
            # Analyze device-specific issues
            if proposal_states[user_id]['device_type'] == 'mobile':
                if len(proposal_states[user_id]['feature_usage']) > 3:
                    pain_points["mobile_feature_overload"] += 1
            
            # Analyze feature adoption
            if len(proposal_states[user_id]['feature_usage']) == 0:
                pain_points["no_feature_usage"] += 1
            elif len(proposal_states[user_id]['feature_usage']) == 1:
                pain_points["limited_feature_usage"] += 1
            
            # Track interaction styles
            if proposal_states[user_id]['revisions'] > 0:
                interaction_styles["revision_focused"] += 1
            if len(proposal_states[user_id]['feature_usage']) > 2:
                interaction_styles["feature_explorer"] += 1
            if current_session_duration > 1800:  # More than 30 minutes
                interaction_styles["extended_engagement"] += 1
            
            # Identify drop-off points
            if proposal_states[user_id]['created'] and not proposal_states[user_id]['time_to_close']:
                if proposal_states[user_id]['revisions'] > 0:
                    drop_off_points["after_revision"] += 1
                else:
                    drop_off_points["after_creation"] += 1
        
        return {
            "pain_points": dict(pain_points),
            "interaction_styles": dict(interaction_styles),
            "drop_off_points": dict(drop_off_points)
        }
    
    def generate_segments(self, analysis_results: Dict) -> List[Dict]:
        """
        Generate user segments based on analysis results
        
        Args:
            analysis_results: Dictionary containing analysis results from analyze_behavior
            
        Returns:
            List of dictionaries, each representing a user segment with:
            - segment_id: Unique identifier for the segment
            - segment_name: Human-readable name for the segment
            - characteristics: List of characteristics that define the segment
            - user_ids: List of user IDs in this segment
            - metrics: Key metrics for this segment
            - recommendations: List of product recommendations for this segment
        """
        segments = []
        user_segments = {}  # Track which segment each user belongs to
        
        # Extract data from analysis results
        user_patterns = analysis_results["user_patterns"]
        action_frequencies = analysis_results["action_frequencies"]
        time_based_patterns = analysis_results["time_based_patterns"]
        metadata_insights = analysis_results["metadata_insights"]
        ux_insights = analysis_results["ux_insights"]
        
        # Define segment types and their characteristics
        segment_definitions = [
            {
                "id": "fast_closers",
                "name": "Fast Closers",
                "criteria": {
                    "interaction_style": "fast_closer",
                    "min_proposals": 1
                },
                "recommendations": [
                    "Pre-fill common clauses based on past successful deals",
                    "Automated deal status updates",
                    "Quick template selection for similar deals"
                ]
            },
            {
                "id": "high_revisers",
                "name": "High Revisers",
                "criteria": {
                    "interaction_style": "high_reviser",
                    "min_revisions": 3
                },
                "recommendations": [
                    "Version control and change tracking",
                    "Collaborative editing features",
                    "Revision history insights"
                ]
            },
            {
                "id": "ghost_users",
                "name": "Ghost After Proposal",
                "criteria": {
                    "interaction_style": "ghost_after_proposal",
                    "min_proposals": 1
                },
                "recommendations": [
                    "Automated follow-up reminders",
                    "Deal status notifications",
                    "Engagement re-engagement campaigns"
                ]
            },
            {
                "id": "power_users",
                "name": "Power Users",
                "criteria": {
                    "min_sessions": 10,
                    "min_features_used": 5,
                    "min_daily_usage": 3,
                    "feature_types": ["advanced"]
                },
                "recommendations": [
                    "Advanced analytics dashboard",
                    "Custom workflow automation",
                    "Bulk operations support"
                ]
            },
            {
                "id": "feature_explorers",
                "name": "Feature Explorers",
                "criteria": {
                    "min_features_tried": 3,
                    "min_feedback": 2,
                    "max_sessions": 5
                },
                "recommendations": [
                    "Feature discovery tours",
                    "Contextual help and tooltips",
                    "Quick start guides"
                ]
            }
        ]
        
        # Process each user and assign them to segments
        for user_id, patterns in user_patterns.items():
            user_metadata = metadata_insights.get(user_id, {})
            time_patterns = time_based_patterns.get(user_id, {})
            ux_patterns = ux_insights.get(user_id, {})
            
            # Calculate user metrics
            user_metrics = self._calculate_user_metrics(
                user_id,
                patterns,
                user_metadata,
                time_patterns,
                ux_patterns
            )
            
            # Check each segment definition
            for segment_def in segment_definitions:
                if self._user_matches_segment(user_metrics, segment_def["criteria"]):
                    # Add user to segment
                    if segment_def["id"] not in user_segments:
                        user_segments[segment_def["id"]] = []
                    user_segments[segment_def["id"]].append(user_id)
        
        # Create final segment objects
        for segment_def in segment_definitions:
            segment_id = segment_def["id"]
            if segment_id in user_segments and user_segments[segment_id]:
                # Calculate segment metrics
                segment_metrics = self._calculate_segment_metrics(
                    user_segments[segment_id],
                    analysis_results
                )
                
                segments.append({
                    "segment_id": segment_id,
                    "segment_name": segment_def["name"],
                    "characteristics": self._get_segment_characteristics(segment_def),
                    "user_ids": user_segments[segment_id],
                    "metrics": segment_metrics,
                    "recommendations": segment_def["recommendations"]
                })
        
        return segments
    
    def _calculate_user_metrics(self, user_id: str, patterns: Dict, metadata: Dict, time_patterns: Dict, ux_patterns: Dict) -> Dict:
        """Calculate metrics for a single user"""
        metrics = {
            "total_actions": time_patterns.get("frequency", 0),
            "session_count": 0,
            "feature_count": 0,
            "feedback_count": 0,
            "daily_usage": 0,
            "features_used": set(),
            "feature_types": set(),
            "primary_time": None,
            "primary_day": None,
            "last_activity": None,
            "interaction_style": ux_patterns.get("interaction_style"),
            "revision_count": ux_patterns.get("revision_count", 0),
            "time_to_close": ux_patterns.get("time_to_close"),
            "drop_off_count": len(ux_patterns.get("drop_off_points", [])),
            "pain_points": ux_patterns.get("pain_points", [])
        }
        
        # Calculate action counts and feature usage
        if "common_actions" in metadata:
            metrics["session_count"] = metadata["common_actions"].get("session_start", 0)
            metrics["feature_count"] = metadata["common_actions"].get("feature_use", 0)
            metrics["feedback_count"] = metadata["common_actions"].get("feedback", 0)
        
        # Calculate feature usage and types
        if "metadata_patterns" in metadata:
            if "feature_name" in metadata["metadata_patterns"]:
                features = metadata["metadata_patterns"]["feature_name"].get("categorical_values", {})
                metrics["features_used"] = set(features.keys())
            
            if "feature_type" in metadata["metadata_patterns"]:
                types = metadata["metadata_patterns"]["feature_type"].get("categorical_values", {})
                metrics["feature_types"] = set(types.keys())
        
        # Calculate time patterns
        if "time_of_day" in time_patterns:
            metrics["primary_time"] = max(
                time_patterns["time_of_day"].items(),
                key=lambda x: x[1]
            )[0]
        
        if "day_of_week" in time_patterns:
            metrics["primary_day"] = max(
                time_patterns["day_of_week"].items(),
                key=lambda x: x[1]
            )[0]
        
        return metrics
    
    def _user_matches_segment(self, user_metrics: Dict, criteria: Dict) -> bool:
        """Check if a user matches a segment's criteria"""
        # Check interaction style
        if "interaction_style" in criteria and user_metrics["interaction_style"] != criteria["interaction_style"]:
            return False
        
        # Check revision criteria
        if "min_revisions" in criteria and user_metrics["revision_count"] < criteria["min_revisions"]:
            return False
        
        # Check proposal criteria
        if "min_proposals" in criteria and user_metrics["session_count"] < criteria["min_proposals"]:
            return False
        
        # Check session criteria
        if "min_sessions" in criteria and user_metrics["session_count"] < criteria["min_sessions"]:
            return False
        if "max_sessions" in criteria and user_metrics["session_count"] > criteria["max_sessions"]:
            return False
        
        # Check feature criteria
        if "min_features_used" in criteria and len(user_metrics["features_used"]) < criteria["min_features_used"]:
            return False
        if "min_features_tried" in criteria and len(user_metrics["features_used"]) < criteria["min_features_tried"]:
            return False
        
        # Check feedback criteria
        if "min_feedback" in criteria and user_metrics["feedback_count"] < criteria["min_feedback"]:
            return False
        
        # Check daily usage
        if "min_daily_usage" in criteria and user_metrics["daily_usage"] < criteria["min_daily_usage"]:
            return False
        
        # Check feature types
        if "feature_types" in criteria:
            if not any(ft in user_metrics["feature_types"] for ft in criteria["feature_types"]):
                return False
        
        # Check time criteria
        if "primary_time" in criteria and user_metrics["primary_time"] != criteria["primary_time"]:
            return False
        
        return True
    
    def _calculate_segment_metrics(self, user_ids: List[str], analysis_results: Dict) -> Dict:
        """Calculate aggregate metrics for a segment"""
        metrics = {
            "total_users": len(user_ids),
            "total_actions": 0,
            "total_sessions": 0,
            "total_features_used": 0,
            "total_feedback": 0,
            "common_features": {},
            "feature_types": {},
            "time_distribution": {
                "morning": 0,
                "afternoon": 0,
                "evening": 0,
                "night": 0
            },
            "interaction_styles": {},
            "revision_stats": {
                "avg_revisions": 0,
                "max_revisions": 0
            },
            "time_to_close_stats": {
                "avg_hours": 0,
                "max_hours": 0
            },
            "drop_off_points": {},
            "common_pain_points": {}
        }
        
        total_revisions = 0
        total_time_to_close = 0
        close_count = 0
        
        for user_id in user_ids:
            user_metadata = analysis_results["metadata_insights"].get(user_id, {})
            time_patterns = analysis_results["time_based_patterns"].get(user_id, {})
            ux_patterns = analysis_results["ux_insights"].get(user_id, {})
            
            # Aggregate action counts
            if "common_actions" in user_metadata:
                metrics["total_sessions"] += user_metadata["common_actions"].get("session_start", 0)
                metrics["total_features_used"] += user_metadata["common_actions"].get("feature_use", 0)
                metrics["total_feedback"] += user_metadata["common_actions"].get("feedback", 0)
            
            # Aggregate features
            if "metadata_patterns" in user_metadata:
                if "feature_name" in user_metadata["metadata_patterns"]:
                    features = user_metadata["metadata_patterns"]["feature_name"].get("categorical_values", {})
                    for feature, count in features.items():
                        metrics["common_features"][feature] = metrics["common_features"].get(feature, 0) + count
                
                if "feature_type" in user_metadata["metadata_patterns"]:
                    types = user_metadata["metadata_patterns"]["feature_type"].get("categorical_values", {})
                    for type_, count in types.items():
                        metrics["feature_types"][type_] = metrics["feature_types"].get(type_, 0) + count
            
            # Aggregate time patterns
            if "time_of_day" in time_patterns:
                for time, count in time_patterns["time_of_day"].items():
                    metrics["time_distribution"][time] += count
            
            # Aggregate UX patterns
            if ux_patterns:
                # Track interaction styles
                style = ux_patterns.get("interaction_style")
                if style:
                    metrics["interaction_styles"][style] = metrics["interaction_styles"].get(style, 0) + 1
                
                # Track revision stats
                revision_count = ux_patterns.get("revision_count", 0)
                total_revisions += revision_count
                metrics["revision_stats"]["max_revisions"] = max(
                    metrics["revision_stats"]["max_revisions"],
                    revision_count
                )
                
                # Track time to close
                time_to_close = ux_patterns.get("time_to_close")
                if time_to_close:
                    total_time_to_close += time_to_close
                    close_count += 1
                    metrics["time_to_close_stats"]["max_hours"] = max(
                        metrics["time_to_close_stats"]["max_hours"],
                        time_to_close
                    )
                
                # Track drop-off points
                for drop_off in ux_patterns.get("drop_off_points", []):
                    action = drop_off["action"]
                    metrics["drop_off_points"][action] = metrics["drop_off_points"].get(action, 0) + 1
                
                # Track pain points
                for pain_point in ux_patterns.get("pain_points", []):
                    metrics["common_pain_points"][pain_point] = metrics["common_pain_points"].get(pain_point, 0) + 1
        
        # Calculate averages
        if total_revisions > 0:
            metrics["revision_stats"]["avg_revisions"] = total_revisions / len(user_ids)
        if close_count > 0:
            metrics["time_to_close_stats"]["avg_hours"] = total_time_to_close / close_count
        
        # Sort features by frequency
        metrics["common_features"] = dict(
            sorted(metrics["common_features"].items(), key=lambda x: x[1], reverse=True)
        )
        
        # Sort feature types by frequency
        metrics["feature_types"] = dict(
            sorted(metrics["feature_types"].items(), key=lambda x: x[1], reverse=True)
        )
        
        # Sort pain points by frequency
        metrics["common_pain_points"] = dict(
            sorted(metrics["common_pain_points"].items(), key=lambda x: x[1], reverse=True)
        )
        
        return metrics
    
    def _get_segment_characteristics(self, segment_def: Dict) -> List[str]:
        """Generate human-readable characteristics for a segment"""
        characteristics = []
        criteria = segment_def["criteria"]
        
        if "interaction_style" in criteria:
            characteristics.append(f"Interaction style: {criteria['interaction_style']}")
        if "min_revisions" in criteria:
            characteristics.append(f"Minimum {criteria['min_revisions']} revisions")
        if "min_proposals" in criteria:
            characteristics.append(f"Minimum {criteria['min_proposals']} proposals")
        if "min_sessions" in criteria:
            characteristics.append(f"Minimum {criteria['min_sessions']} sessions")
        if "max_sessions" in criteria:
            characteristics.append(f"Maximum {criteria['max_sessions']} sessions")
        if "min_features_used" in criteria:
            characteristics.append(f"Minimum {criteria['min_features_used']} features used")
        if "min_features_tried" in criteria:
            characteristics.append(f"Minimum {criteria['min_features_tried']} features tried")
        if "min_feedback" in criteria:
            characteristics.append(f"Minimum {criteria['min_feedback']} feedback submissions")
        if "min_daily_usage" in criteria:
            characteristics.append(f"Minimum {criteria['min_daily_usage']} daily usage")
        if "feature_types" in criteria:
            characteristics.append(f"Feature types: {', '.join(criteria['feature_types'])}")
        if "primary_time" in criteria:
            characteristics.append(f"Most active during {criteria['primary_time']}")
        
        return characteristics
    
    def get_recommendations(self, segment: Dict) -> List[str]:
        """Generate personalized recommendations based on segment characteristics"""
        recommendations = []
        
        # Get segment metrics and characteristics
        metrics = segment.get('metrics', {})
        pain_points = metrics.get('common_pain_points', {})
        interaction_styles = metrics.get('interaction_styles', {})
        feature_usage = metrics.get('feature_usage', {})
        time_patterns = metrics.get('time_patterns', {})
        revision_stats = metrics.get('revision_stats', {})
        time_to_close_stats = metrics.get('time_to_close_stats', {})
        drop_off_points = metrics.get('drop_off_points', {})
        
        # Recommendations based on pain points
        if pain_points.get('short_session_duration', 0) > 0:
            recommendations.append("Implement quick-start guides and tooltips for faster onboarding")
        if pain_points.get('quick_feature_abandonment', 0) > 0:
            recommendations.append("Add interactive tutorials for complex features")
        if pain_points.get('excessive_revisions', 0) > 0:
            recommendations.append("Introduce template suggestions based on common revision patterns")
        if pain_points.get('slow_proposal_closure', 0) > 0:
            recommendations.append("Implement automated follow-up reminders for pending proposals")
        if pain_points.get('quick_proposal_views', 0) > 0:
            recommendations.append("Add proposal preview summaries for quick scanning")
        if pain_points.get('extended_proposal_views', 0) > 0:
            recommendations.append("Implement section navigation and search within proposals")
        if pain_points.get('low_ratings', 0) > 0:
            recommendations.append("Add contextual help and support options at pain points")
        if pain_points.get('minimal_feedback', 0) > 0:
            recommendations.append("Implement structured feedback forms with specific questions")
        if pain_points.get('extended_sessions', 0) > 0:
            recommendations.append("Add session progress indicators and break reminders")
        if pain_points.get('mobile_feature_overload', 0) > 0:
            recommendations.append("Optimize mobile interface with progressive feature disclosure")
        if pain_points.get('no_feature_usage', 0) > 0:
            recommendations.append("Implement feature discovery tours and usage incentives")
        if pain_points.get('limited_feature_usage', 0) > 0:
            recommendations.append("Add personalized feature recommendations based on usage patterns")
        
        # Recommendations based on interaction styles
        if interaction_styles.get('revision_focused', 0) > 0:
            recommendations.append("Implement version control and change tracking for proposals")
        if interaction_styles.get('feature_explorer', 0) > 0:
            recommendations.append("Add advanced feature discovery and customization options")
        if interaction_styles.get('extended_engagement', 0) > 0:
            recommendations.append("Implement session persistence and auto-save features")
        
        # Recommendations based on drop-off points
        if drop_off_points.get('after_creation', 0) > 0:
            recommendations.append("Add post-creation engagement prompts and next-step suggestions")
        if drop_off_points.get('after_revision', 0) > 0:
            recommendations.append("Implement revision completion checklists and validation")
        
        # Recommendations based on time patterns
        if time_patterns.get('morning_users', 0) > 0:
            recommendations.append("Schedule important notifications and updates for morning hours")
        if time_patterns.get('evening_users', 0) > 0:
            recommendations.append("Implement night mode and reduced notification frequency")
        
        # Recommendations based on feature usage
        if feature_usage.get('advanced_features', 0) > 0:
            recommendations.append("Add power user shortcuts and advanced customization options")
        if feature_usage.get('basic_features', 0) > 0:
            recommendations.append("Implement progressive feature discovery and guided tours")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations

def main():
    # Example usage
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    agent = BehavioralSegmentationAgent(api_key)
    # Add example usage here

if __name__ == "__main__":
    main() 