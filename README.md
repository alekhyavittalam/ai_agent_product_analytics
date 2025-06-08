# Behavioral Segmentation Agent

A tool for analyzing user behaviors and generating meaningful segments based on interaction patterns, with a modern web interface for visualization.

## Project Structure

```
.
├── backend/
│   ├── behavioral_segmentation_agent.py  # Core segmentation logic
│   └── app.py                           # Flask backend server
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css               # Custom styles
│   │   └── js/
│   │       └── main.js                 # Frontend logic
│   └── templates/
│       └── index.html                  # Main dashboard template
└── README.md
```

## Features

- User behavior analysis and segmentation
- Real-time visualization of segment data
- Interactive dashboard with key metrics
- Pain point identification
- Personalized recommendations
- Modern, responsive UI

## Setup Instructions

1. Install dependencies:
```bash
pip install flask python-dotenv
```

2. Set up environment variables:
Create a `.env` file in the root directory with:
```
OPENROUTER_API_KEY=your_api_key_here
```

3. Run the application:
```bash
cd backend
python app.py
```

4. Access the dashboard:
Open your browser and navigate to `http://localhost:5000`

## Usage

The dashboard will automatically:
- Generate sample user behaviors
- Analyze interaction patterns
- Create meaningful segments
- Display the top 3 segments with the most users
- Show key metrics and recommendations for each segment

## Development

- Backend: Python/Flask
- Frontend: HTML, Tailwind CSS, JavaScript
- Dependencies: Flask, python-dotenv

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request