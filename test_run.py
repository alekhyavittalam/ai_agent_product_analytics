import os
from datetime import datetime, timedelta
from behavioral_segmentation_agent import BehavioralSegmentationAgent, UserBehavior
import json
import random

def create_sample_behaviors():
    """Create sample user behaviors for testing with more diverse patterns"""
    now = datetime.now()
    behaviors = []
    
    # Helper function to create proposal-related behaviors
    def create_proposal_behaviors(user_id, base_time, proposal_id, value, revision_count=0, is_ghost=False):
        proposal_behaviors = [
            UserBehavior(
                user_id=user_id,
                timestamp=base_time,
                action="proposal_create",
                metadata={"proposal_id": proposal_id, "type": "investment", "initial_value": value}
            )
        ]
        
        if not is_ghost:
            # Add revisions if any
            for i in range(revision_count):
                proposal_behaviors.append(
                    UserBehavior(
                        user_id=user_id,
                        timestamp=base_time + timedelta(hours=i+1),
                        action="proposal_revision",
                        metadata={"proposal_id": proposal_id, "revision_number": i+1, "changes": ["updated_terms"]}
                    )
                )
            
            # Add proposal close
            proposal_behaviors.append(
                UserBehavior(
                    user_id=user_id,
                    timestamp=base_time + timedelta(hours=revision_count+1),
                    action="proposal_close",
                    metadata={"proposal_id": proposal_id, "status": "accepted", "final_value": value}
                )
            )
        else:
            # Add just a view for ghost users
            proposal_behaviors.append(
                UserBehavior(
                    user_id=user_id,
                    timestamp=base_time + timedelta(minutes=30),
                    action="proposal_view",
                    metadata={"proposal_id": proposal_id, "view_duration": 300}
                )
            )
        
        return proposal_behaviors

    # Create 20 users with different patterns
    for i in range(1, 21):
        user_id = f"user{i}"
        base_time = now - timedelta(days=random.randint(1, 30))
        
        # Randomly assign user type
        user_type = random.choice(['fast_closer', 'high_reviser', 'ghost', 'power_user', 'feature_explorer'])
        
        if user_type == 'fast_closer':
            # Create 2-3 fast-closing proposals
            for j in range(random.randint(2, 3)):
                behaviors.extend(create_proposal_behaviors(
                    user_id,
                    base_time - timedelta(days=j*2),
                    f"P{i}{j}",
                    random.randint(500000, 2000000),
                    revision_count=0
                ))
        
        elif user_type == 'high_reviser':
            # Create 1-2 proposals with many revisions
            for j in range(random.randint(1, 2)):
                behaviors.extend(create_proposal_behaviors(
                    user_id,
                    base_time - timedelta(days=j*3),
                    f"P{i}{j}",
                    random.randint(300000, 1500000),
                    revision_count=random.randint(3, 5)
                ))
        
        elif user_type == 'ghost':
            # Create 1-2 proposals that are viewed but not closed
            for j in range(random.randint(1, 2)):
                behaviors.extend(create_proposal_behaviors(
                    user_id,
                    base_time - timedelta(days=j*2),
                    f"P{i}{j}",
                    random.randint(200000, 1000000),
                    is_ghost=True
                ))
        
        elif user_type == 'power_user':
            # Create advanced feature usage and proposals
            behaviors.extend([
                UserBehavior(
                    user_id=user_id,
                    timestamp=base_time,
                    action="session_start",
                    metadata={"session_id": f"S{i}", "device": "desktop", "feature_type": "advanced"}
                ),
                UserBehavior(
                    user_id=user_id,
                    timestamp=base_time + timedelta(minutes=5),
                    action="feature_use",
                    metadata={"feature_name": "advanced_analytics", "feature_type": "advanced", "duration": 45}
                ),
                UserBehavior(
                    user_id=user_id,
                    timestamp=base_time + timedelta(minutes=30),
                    action="feature_use",
                    metadata={"feature_name": "custom_dashboard", "feature_type": "advanced", "duration": 30}
                )
            ])
            # Add 1-2 proposals
            for j in range(random.randint(1, 2)):
                behaviors.extend(create_proposal_behaviors(
                    user_id,
                    base_time + timedelta(days=j),
                    f"P{i}{j}",
                    random.randint(1000000, 3000000),
                    revision_count=random.randint(0, 2)
                ))
        
        else:  # feature_explorer
            # Create basic feature usage and feedback
            behaviors.extend([
                UserBehavior(
                    user_id=user_id,
                    timestamp=base_time,
                    action="session_start",
                    metadata={"session_id": f"S{i}", "device": "mobile", "feature_type": "core"}
                ),
                UserBehavior(
                    user_id=user_id,
                    timestamp=base_time + timedelta(minutes=5),
                    action="feature_use",
                    metadata={"feature_name": "basic_analytics", "feature_type": "core", "duration": 15}
                ),
                UserBehavior(
                    user_id=user_id,
                    timestamp=base_time + timedelta(minutes=25),
                    action="feedback",
                    metadata={"feature_name": "basic_analytics", "rating": random.randint(3, 5), "comment": "Easy to use"}
                )
            ])
            # Add 1 proposal
            behaviors.extend(create_proposal_behaviors(
                user_id,
                base_time + timedelta(days=1),
                f"P{i}0",
                random.randint(200000, 800000),
                revision_count=random.randint(0, 1)
            ))
    
    return behaviors

def print_top_segments(segments):
    """Print information about the top 3 segments with the most users"""
    print("\n=== Top 3 User Segments ===")
    
    # Sort segments by number of users
    sorted_segments = sorted(segments, key=lambda x: len(x['user_ids']), reverse=True)
    top_segments = sorted_segments[:3]
    
    for segment in top_segments:
        print(f"\n{segment['segment_name']} (ID: {segment['segment_id']})")
        print(f"Number of Users: {len(segment['user_ids'])}")
        
        print("\nKey Characteristics:")
        for char in segment['characteristics']:
            print(f"  - {char}")
        
        print("\nKey Metrics:")
        metrics = segment['metrics']
        print(f"  - Total Actions: {metrics['total_actions']}")
        print(f"  - Average Revisions: {metrics['revision_stats']['avg_revisions']:.2f}")
        print(f"  - Average Time to Close: {metrics['time_to_close_stats']['avg_hours']:.2f} hours")
        
        print("\nCommon Pain Points:")
        for pain_point, count in metrics['common_pain_points'].items():
            print(f"  - {pain_point}: {count} times")
        
        print("\nTop Recommendations:")
        recommendations = agent.get_recommendations(segment)
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5 recommendations
            print(f"  {i}. {rec}")

def main():
    # Get API key from environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set the OPENROUTER_API_KEY environment variable")
        return

    # Initialize the agent
    global agent
    agent = BehavioralSegmentationAgent(api_key)
    
    # Create sample behaviors
    behaviors = create_sample_behaviors()
    
    try:
        # Test behavior analysis
        print("Analyzing behaviors...")
        analysis_results = agent.analyze_behavior(behaviors)
        
        # Test segment generation
        print("\nGenerating segments...")
        segments = agent.generate_segments(analysis_results)
        
        # Print top segments information
        print_top_segments(segments)
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    main() 