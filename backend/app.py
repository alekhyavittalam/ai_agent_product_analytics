from flask import Flask, render_template, jsonify, request
from behavioral_segmentation_agent import BehavioralSegmentationAgent, UserBehavior
from datetime import datetime, timedelta
import random
import os
import pandas as pd
from werkzeug.utils import secure_filename
import json

# Initialize Flask app with the frontend directory
app = Flask(__name__, 
    template_folder='../frontend/templates',
    static_folder='../frontend/static'
)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_csv_to_behaviors(csv_path):
    """Convert CSV data to UserBehavior objects"""
    df = pd.read_csv(csv_path)
    behaviors = []
    
    # Expected columns: user_id, timestamp, action, metadata
    for _, row in df.iterrows():
        # Parse timestamp
        try:
            timestamp = pd.to_datetime(row['timestamp'])
        except:
            timestamp = datetime.now()
        
        # Parse metadata
        try:
            metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else {}
        except:
            metadata = {}
        
        behavior = UserBehavior(
            user_id=str(row['user_id']),
            timestamp=timestamp,
            action=str(row['action']),
            metadata=metadata
        )
        behaviors.append(behavior)
    
    return behaviors

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Parse CSV and create behaviors
            behaviors = parse_csv_to_behaviors(filepath)
            
            # Initialize the agent
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                return jsonify({"error": "API key not found"}), 500
            
            agent = BehavioralSegmentationAgent(api_key)
            
            # Analyze behaviors
            analysis_results = agent.analyze_behavior(behaviors)
            
            # Generate segments
            segments = agent.generate_segments(analysis_results)
            
            # Sort segments by number of users
            sorted_segments = sorted(segments, key=lambda x: len(x['user_ids']), reverse=True)
            top_segments = sorted_segments[:3]
            
            # Format segments for frontend
            formatted_segments = []
            for segment in top_segments:
                formatted_segment = {
                    'name': segment['segment_name'],
                    'id': segment['segment_id'],
                    'user_count': len(segment['user_ids']),
                    'characteristics': segment['characteristics'],
                    'metrics': {
                        'total_actions': segment['metrics']['total_actions'],
                        'avg_revisions': round(segment['metrics']['revision_stats']['avg_revisions'], 2),
                        'avg_time_to_close': round(segment['metrics']['time_to_close_stats']['avg_hours'], 2)
                    },
                    'pain_points': [
                        {'name': point, 'count': count}
                        for point, count in segment['metrics']['common_pain_points'].items()
                    ],
                    'recommendations': agent.get_recommendations(segment)[:5]  # Top 5 recommendations
                }
                formatted_segments.append(formatted_segment)
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            return jsonify({
                'segments': formatted_segments,
                'total_users': sum(len(segment['user_ids']) for segment in top_segments)
            })
            
        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/segments')
def get_segments():
    # Initialize the agent
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return jsonify({"error": "API key not found"}), 500
    
    agent = BehavioralSegmentationAgent(api_key)
    
    try:
        # Create sample behaviors
        behaviors = create_sample_behaviors()
        
        # Analyze behaviors
        analysis_results = agent.analyze_behavior(behaviors)
        
        # Generate segments
        segments = agent.generate_segments(analysis_results)
        
        # Sort segments by number of users
        sorted_segments = sorted(segments, key=lambda x: len(x['user_ids']), reverse=True)
        top_segments = sorted_segments[:3]
        
        # Format segments for frontend
        formatted_segments = []
        for segment in top_segments:
            formatted_segment = {
                'name': segment['segment_name'],
                'id': segment['segment_id'],
                'user_count': len(segment['user_ids']),
                'characteristics': segment['characteristics'],
                'metrics': {
                    'total_actions': segment['metrics']['total_actions'],
                    'avg_revisions': round(segment['metrics']['revision_stats']['avg_revisions'], 2),
                    'avg_time_to_close': round(segment['metrics']['time_to_close_stats']['avg_hours'], 2)
                },
                'pain_points': [
                    {'name': point, 'count': count}
                    for point, count in segment['metrics']['common_pain_points'].items()
                ],
                'recommendations': agent.get_recommendations(segment)[:5]  # Top 5 recommendations
            }
            formatted_segments.append(formatted_segment)
        
        return jsonify({
            'segments': formatted_segments,
            'total_users': sum(len(segment['user_ids']) for segment in top_segments)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 