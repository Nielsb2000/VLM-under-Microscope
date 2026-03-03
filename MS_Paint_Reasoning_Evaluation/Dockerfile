# Dockerfile for Plotly Dash dashboard in MS_Paint_Reasoning_Evaluation
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-dashboard.txt ./
RUN pip install --no-cache-dir -r requirements-dashboard.txt


# Copy dashboard app code and dependencies
COPY dashboard.py ./
COPY json_results_to_df.py ./
COPY Results ./Results

# Expose Dash port
EXPOSE 8050

CMD ["python", "dashboard.py"]
