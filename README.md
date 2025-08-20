# Homework 2: MLOps and Deployment - Bedrock Chatbot

This project extends the Homework 1 AWS Bedrock-based chatbot by splitting it into a **FastAPI backend** and a **Streamlit frontend**, containerizing both with Docker, orchestrating them with Docker Compose, and deploying them to an AWS EC2 instance using a CI/CD pipeline with a self-hosted GitHub Actions runner. The backend leverages AWS Bedrock's Claude 3.5 Sonnet model for natural language inference, while the frontend provides an interactive web interface for querying the chatbot and visualizing responses.

## Project Structure

The project is organized as follows, reflecting the separation of backend and frontend services:

```plaintext
.
├── backend/
│   ├── Dockerfile           # Backend Dockerfile
│   ├── requirements.txt     # Backend dependencies
│   ├── app.py               # Backend code
│   └── .env                 # Example environment variables
├── frontend/
│   ├── app.py               # Streamlit application
│   ├── Dockerfile           # Frontend Dockerfile
│   ├── requirements.txt     # Frontend dependencies
│   └── .env.example         # Example environment variables
├── assets/
│   ├── screenshots/         # Screenshots of frontend and backend
├── .github/
│   └── workflows/
│       └── ci-build.yaml    # GitHub Actions workflow
├── docker-compose.yml       # Docker Compose configuration
├── README.md                # This file
└── .gitignore               # Git ignore file
```

- **Backend (`backend/`)**: Contains the FastAPI API for querying the AWS Bedrock knowledge base.
- **Frontend (`frontend/`)**: Includes the Streamlit web interface for user input and response visualization.
- **Assets (`assets/`)**: Stores screenshots.
- **Workflows (`.github/workflows/`)**: Defines the CI/CD pipeline.

## Prerequisites

To run this project locally or deploy it to AWS EC2, you’ll need:

- **Docker** and **Docker Compose** (for containerization)
- **AWS Account** with access to:
  - Bedrock (Claude 3.5 Sonnet model: `anthropic.claude-3-5-sonnet-20240620-v1:0`)
  - Knowledge base (ID: `JGMPKF6VEI`)
  - EC2 instance (for deployment)
- **GitHub Repository** with SSH access configured
- **Python 3.9+** (for local development)
- **GitHub Actions Self-Hosted Runner** (set up on EC2)
- AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)

## Setup Instructions

### Local Setup

1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Set Up Environment Variables**:
   - Copy the example environment files:
     ```bash
     cp backend/.env.example backend/.env
     cp frontend/.env.example frontend/.env
     ```
   - Update `backend/.env` with your AWS credentials and configuration:
     ```
     AWS_ACCESS_KEY_ID=<your-access-key>
     AWS_SECRET_ACCESS_KEY=<your-secret-key>
     AWS_REGION=us-east-1
     KNOWLEDGE_BASE_ID=JGMPKF6VEI
     CLAUDE_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
     ```
   - Update `frontend/.env` with the backend URL:
     ```
     BACKEND_URL=http://backend:8000
     ```

3. **Build and Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```
   - Access the FastAPI backend at `http://localhost:8000/docs`.
   - Access the Streamlit frontend at `http://localhost:8501`.

4. **Test the Application**:
   - Open `http://localhost:8501` in your browser.
   - Enter a query (e.g., "Hello, can you help me?") in the Streamlit interface.
   - View the response and visualizations (e.g., timestamp timeline, citation counts).

### AWS EC2 Deployment Setup

1. **Launch an EC2 Instance**:
   - Use an Ubuntu 20.04+ AMI with at least 4GB RAM (e.g., `t2.medium`).
   - Configure security groups to allow:
     - Port 22 (SSH) for GitHub Actions.
     - Port 8000 (FastAPI backend).
     - Port 8501 (Streamlit frontend).
   - Attach an IAM role with Bedrock permissions.

2. **Set Up SSH Access**:
   - Generate an SSH key pair:
     ```bash
     ssh-keygen -t rsa -b 4096 -f ~/.ssh/github-actions-key
     ```
   - Add the public key (`github-actions-key.pub`) to `~/.ssh/authorized_keys` on the EC2 instance.
   - Add the private key as a secret (`EC2_SSH_KEY`) in your GitHub repository (Settings > Secrets and variables > Actions).

3. **Install Dependencies on EC2**:
   - SSH into the EC2 instance:
     ```bash
     ssh -i <your-key.pem> ubuntu@<ec2-public-ip>
     ```
   - Install Docker and Docker Compose:
     ```bash
     sudo apt update
     sudo apt install -y docker.io docker-compose
     sudo usermod -aG docker ubuntu
     ```

4. **Clone the Repository on EC2**:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

5. **Set Up GitHub Actions Self-Hosted Runner**:
   - Follow GitHub’s instructions to add a self-hosted runner (Settings > Actions > Runners > New self-hosted runner).
   - On the EC2 instance, configure the runner:
     ```bash
     mkdir actions-runner && cd actions-runner
     curl -o actions-runner-linux-x64-<version>.tar.gz -L https://github.com/actions/runner/releases/download/v<version>/actions-runner-linux-x64-<version>.tar.gz
     tar xzf ./actions-runner-linux-x64-<version>.tar.gz
     ./config.sh --url https://github.com/<your-username>/<your-repo> --token <your-runner-token>
     ./run.sh
     ```
   - Run the runner as a service.

6. **Set Up Environment Variables on EC2**:
   - Copy `.env` files to `backend/` and `frontend/` directories on EC2.
   - Update with the same values as in the local setup.

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) automates building, testing, and deploying the services to EC2.

### Workflow Overview
- **Triggers**: Runs on `push` or `pull_request` to the `main` branch.
- **Jobs**:
  - **Lint**: Runs `ruff` to check code quality.
  - **Build**: Builds Docker images for backend and frontend.
  - **Deploy**: Deploys to EC2 via SSH using the self-hosted runner.

### Example Workflow
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install ruff
      - name: Run ruff
        run: |
          ruff check backend/src frontend/src

  build:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker images
        run: |
          docker-compose build

  deploy:
    runs-on: self-hosted
    needs: build
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to EC2
        env:
          EC2_SSH_KEY: ${{ secrets.EC2_SSH_KEY }}
          EC2_HOST: <your-ec2-public-ip>
          EC2_USER: ubuntu
        run: |
          echo "$EC2_SSH_KEY" > ssh_key
          chmod 600 ssh_key
          ssh -i ssh_key $EC2_USER@$EC2_HOST << 'EOF'
            cd <your-repo-name>
            git pull origin main
            docker-compose down
            docker-compose up -d --build
          EOF
          rm ssh_key
```

## Backend Details

- **Framework**: FastAPI
- **Endpoint**: `/predict` (POST)
  - Accepts a JSON payload with a `message` field (and optional `session_id`).
  - Queries the AWS Bedrock knowledge base using Claude 3.5 Sonnet.
  - Returns the response with answer, session ID, citations, and timestamp.
- **Model**: Uses the Claude 3.5 Sonnet model (`anthropic.claude-3-5-sonnet-20240620-v1:0`) via AWS Bedrock.

**Dockerfile** (`backend/Dockerfile`):
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/src ./src
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Frontend Details

- **Framework**: Streamlit
- **Interface**: A web UI where users input queries, send them to the `/predict` endpoint, and view responses.
- **Visualizations**:
  - Displays the response text.
  - Shows a timeline of response timestamps using Plotly.
  - Visualizes citation counts (if any) in a bar chart.

**Dockerfile** (`frontend/Dockerfile`):
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY frontend/src ./src
ENV PYTHONPATH=/app/src
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Docker Compose Configuration

The `docker-compose.yml` file orchestrates the backend and frontend services.

```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    env_file: ./backend/.env
    ports:
      - "8000:8000"
    networks:
      - app-network

  frontend:
    build: ./frontend
    env_file: ./frontend/.env
    ports:
      - "8501:8501"
    depends_on:
      - backend
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
```

## Improvements and Creativity

- **Backend**:
  - Enhanced the `/predict` endpoint with session management for conversational context.
  - Added detailed error handling for AWS Bedrock API calls.
  - Used Pydantic for request/response validation.
- **Frontend**:
  - Designed a clean Streamlit UI with a chat-like layout.
  - Added Plotly visualizations for response metadata (timestamp timeline, citation counts).
  - Included a session ID display for user convenience.
- **CI/CD**:
  - Integrated `ruff` for linting to enforce code quality.
  - Automated deployment to EC2 with zero-downtime updates using `docker-compose`.
- **Assets**:
  - Added screenshots of the Streamlit UI and FastAPI Swagger docs in `assets/screenshots/`.

## Testing

1. **Local Testing**:
   - Run `docker-compose up --build`.
   - Test the backend: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"message": "Hello"}'`.
   - Test the frontend: Open `http://localhost:8501` and submit a query.

2. **EC2 Testing**:
   - SSH into the EC2 instance and run `docker-compose up -d`.
   - Access the frontend at `http://<ec2-public-ip>:8501`.

## Troubleshooting

- **AWS Bedrock Errors**:
  - Ensure `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `KNOWLEDGE_BASE_ID`, and `CLAUDE_MODEL_ID` are correct.
  - Contact your AWS administrator if `AccessDeniedException` or `ResourceNotFoundException` occurs.
- **Docker Issues**:
  - Check Docker logs: `docker-compose logs`.
  - Ensure ports 8000 and 8501 are open in the EC2 security group.
- **GitHub Actions**:
  - Verify the self-hosted runner is online in GitHub Settings > Actions > Runners.
  - Check SSH key configuration if deployment fails.

## Screenshots

- **Frontend UI**: See `assets/screenshots/frontend.png` for the Streamlit interface.
- **Backend API**: See `assets/screenshots/backend.png` for the FastAPI Swagger docs.

## License

MIT License