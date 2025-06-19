# Shopping Website Behavioral Analysis Dashboard

## Overview
This Streamlit application provides a powerful tool for analyzing user behavior on a shopping website. It leverages clustering techniques (UMAP + HDBSCAN) and AI-powered insights to identify user segments, pinpoint drop-off points in the shopping journey, and generate actionable recommendations with testable hypotheses.

## Features
- **Behavioral Clustering**: Automatically segments users based on their interaction patterns, visualized with interactive UMAP plots where data point size scales with cluster size.
- **User Journey Funnel**: Visualizes the user's progress through key stages of the shopping flow (e.g., Product View, Search, Filter, Add to Cart, Remove from Cart, Abandon Cart, Checkout Start, Purchase Complete) to identify bottlenecks.
- **AI-Powered Insights**: Generates comprehensive insights, including cluster names, personas, metric summaries, pain points, recommendations, and A/B test hypotheses, with customizable output modes (Product Manager, Technical) and tones (UX Designer, Developer, Business).
- **A/B Test Simulation**: Allows for quick simulation of A/B test configurations for each recommendation, providing mock metrics like expected improvement, sample size, and duration.
- **Markdown Report Generation**: Export detailed analysis reports in Markdown format, including cluster overviews, KPIs, funnel data, pain points, hypotheses, and recommendations.
- **Modern UI/UX**: Clean, aesthetic dark-mode interface with a guided landing page for easy onboarding.

## Getting Started

### Prerequisites
- Python 3.8+
- An OpenAI API Key (or OpenRouter API Key compatible with OpenAI's API)

### Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/your-username/behavioral_segmentation_agent.git
    cd behavioral_segmentation_agent
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration (API Key)
The application requires an OpenAI API key to generate AI insights. Set it as an environment variable:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
# If using OpenRouter, you might also need:
export OPENROUTER_API_KEY="your_openrouter_api_key_here" # The app is configured to use OpenRouter's base URL if an OpenAI key is provided.
```
Replace `"your_openai_api_key_here"` with your actual API key.

### Running the Application

Once the dependencies are installed and the API key is set, run the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your web browser, usually at `http://localhost:8501`.

## Usage

1.  **Welcome Page**: Upon launching, you'll see a welcome page guiding you through the tool's capabilities and data requirements.
2.  **Upload Data**: Use the "Upload User Behavior CSV" file uploader in the sidebar to upload your behavioral data. A `sample_behavior_data.csv` can be downloaded from the landing page for demonstration.
    **CSV Format Requirements**:
    Your CSV must contain the following columns:
    -   `user_id`: Unique identifier for each user.
    -   `timestamp`: ISO format timestamps (e.g., `2024-03-20T10:00:00`).
    -   `action`: The type of user interaction (e.g., `product_view`, `add_to_cart`, `purchase_complete`).
    -   `metadata`: A JSON string containing additional details about the action (e.g., `{"product_id": "A1"}`, `{"cart_value": 100}`).
3.  **Explore Dashboard**: After uploading, the dashboard will display:
    -   **User Clusters**: An interactive scatter plot visualizing user clusters based on behavioral patterns.
    -   **Cluster Selection**: A dropdown to select individual clusters for detailed analysis.
    -   **KPIs**: Key Performance Indicators like Conversion Rate and Completion Rate.
    -   **User Journey Funnel**: A visual representation of user flow and drop-off points.
    -   **AI Insights**: LLM-generated cluster names, personas, metric summaries, pain points, recommendations, and hypotheses.
    -   **A/B Test Simulation**: Buttons to simulate A/B tests for each recommendation.
4.  **Generate Report**: Click the "Generate Report" button to create a comprehensive Markdown report of the selected cluster's analysis, which can be downloaded.

## Sample Data
A sample CSV with required columns is available for download directly from the application's landing page, or you can find a similar structure in `sample_data.csv` within the repository.

## Contributing
If you'd like to contribute, please fork the repository and submit a pull request.

## License
This project is open-source and available under the [MIT License](LICENSE).