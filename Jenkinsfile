// Publishes the final SAC model to a Gitea release after training completes.
// Runs on the node that has the trained model under ML/models (the GPU trainer, VM 203).
// Credentials (configure in Jenkins):
//   - gitea-token : Secret text -> exposed as GITEA_TOKEN
pipeline {
    agent { label params.AGENT_LABEL ?: 'ml-gpu' }
    parameters {
        string(name: 'AGENT_LABEL', defaultValue: 'ml-gpu', description: 'Node label of the trainer that holds ML/models')
        string(name: 'ML_DIR',      defaultValue: '/home/user/ML', description: 'ML checkout dir on the trainer')
        string(name: 'RELEASE_TAG', defaultValue: '', description: 'Optional release tag (default: date-based)')
    }
    options { timeout(time: 30, unit: 'MINUTES') }
    stages {
        stage('Wait for training to finish') {
            steps {
                sh '''
                  test -f "${ML_DIR}/models/sac_smart_charger_final.zip" \
                    || { echo "Final model not present yet — training still running. Aborting."; exit 1; }
                '''
            }
        }
        stage('Publish model to Gitea release') {
            steps {
                withCredentials([string(credentialsId: 'gitea-token', variable: 'GITEA_TOKEN')]) {
                    sh '''
                      cd "${ML_DIR}"
                      RELEASE_TAG="${RELEASE_TAG:-sac-model-$(date +%Y%m%d-%H%M)}" \
                        bash ci/publish_model.sh
                    '''
                }
            }
        }
    }
}
