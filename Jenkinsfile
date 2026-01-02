pipeline {
    agent any
    environment { 
        DOCKER_IMAGE = "your-dockerhub-username/iris-ml-api"
        DOCKER_TAG = "${BUILD_NUMBER}"
        DOCKER_CREDENTIALS_ID = "dockerhub-credentials" 
    }

    stages { 
        stage('Checkout') {
            steps { 
                echo "Checking out code from GitHub..."
                checkout scm 
            }
        }

        stage("Setup Python Environment") {
            steps { 
                echo "Setting up Python environment..."
                sh '''
                    python3 -m venv venv 
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r src/requirements.txt
                '''
            }
        }
        stage("Train Model") {
            steps { echo "Training ML model..." 
                sh '''
                    . venv/bin/activate
                    cd src
                    python train_model.py
                    cd ..
                '''
            }
        }
    }
}
