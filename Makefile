GIT_ROOT := $(shell git rev-parse --show-toplevel)
INFRA_ROOT := $(GIT_ROOT)/infra
VERSION ?= local
RELEASE_VERSION ?= latest
TF_INFRA_ROOT := $(INFRA_ROOT)
ECR_REPO_BASE_URL = 626635410672.dkr.ecr.us-east-1.amazonaws.com
WHISPER_RAY_SERVICE_REPO_URL = $(ECR_REPO_BASE_URL)/whisper-ray-service
EKS_CLUSTER_NAME := kuberay-cluster
REGION = us-east-1

run-client:
	streamlit run app.py

ecr-login:
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(ECR_REPO_BASE_URL)

set-jarvis-kubectl-context:
	aws eks update-kubeconfig --region ${REGION} --name ${EKS_CLUSTER_NAME}

build-ray-service:
	docker build --platform linux/amd64 --no-cache -t whisper-ray-service:$(RELEASE_VERSION) .

push-docker-backend-service:
	ecr-login
	docker tag whisper-ray-service:$(RELEASE_VERSION) $(WHISPER_RAY_SERVICE_REPO_URL):$(RELEASE_VERSION)
	docker push $(WHISPER_RAY_SERVICE_REPO_URL):$(RELEASE_VERSION)

build-push-ray-service: build-ray-service push-docker-backend-service

deploy-ray-service:
	kubectl apply -f $(GIT_ROOT)/Whisper-RayService.yaml