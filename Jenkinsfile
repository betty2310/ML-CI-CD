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
    }
}
