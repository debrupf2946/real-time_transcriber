GIT_ROOT := $(shell git rev-parse --show-toplevel)
INFRA_ROOT := $(GIT_ROOT)/infra
VERSION ?= local
RELEASE_VERSION ?= latest
TF_INFRA_ROOT := $(INFRA_ROOT)
<<<<<<< HEAD
ECR_REPO_BASE_URL = 975226449092.dkr.ecr.us-east-1.amazonaws.com
=======
ECR_REPO_BASE_URL = 626635410672.dkr.ecr.us-east-1.amazonaws.com
>>>>>>> 464fb62cfab5620cbc8317a40499440a24f3a356
WHISPER_RAY_SERVICE_REPO_URL = $(ECR_REPO_BASE_URL)/whisper-ray-service
EKS_CLUSTER_NAME := kuberay-cluster
REGION = us-east-1

<<<<<<< HEAD
=======
run-client:
	streamlit run app.py

>>>>>>> 464fb62cfab5620cbc8317a40499440a24f3a356
ecr-login:
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(ECR_REPO_BASE_URL)

set-jarvis-kubectl-context:
<<<<<<< HEAD
	aws eks update-kubeconfig --region $(REGION) --name $(EKS_CLUSTER_NAME)
=======
	aws eks update-kubeconfig --region ${REGION} --name ${EKS_CLUSTER_NAME}
>>>>>>> 464fb62cfab5620cbc8317a40499440a24f3a356

build-ray-service:
	docker build --platform linux/amd64 --no-cache -t whisper-ray-service:$(RELEASE_VERSION) .

push-docker-whisper-ray-service:
<<<<<<< HEAD
	docker tag whisper-ray-service:$(RELEASE_VERSION) $(WHISPER_RAY_SERVICE_REPO_URL):$(RELEASE_VERSION)
=======
	docker tag whisper-ray-service:$(VERSION) $(WHISPER_RAY_SERVICE_REPO_URL):$(RELEASE_VERSION)
>>>>>>> 464fb62cfab5620cbc8317a40499440a24f3a356
	docker push $(WHISPER_RAY_SERVICE_REPO_URL):$(RELEASE_VERSION)

build-push-ray-service: build-ray-service push-docker-whisper-ray-service

deploy-ray-service:
	kubectl apply -f $(GIT_ROOT)/Whisper-RayService.yaml

undeploy-ray-service:
<<<<<<< HEAD
	kubectl delete -f $(GIT_ROOT)/Whisper-RayService.yaml
=======
	kubectl delete -f $(GIT_ROOT)/Whisper-RayService.yaml
>>>>>>> 464fb62cfab5620cbc8317a40499440a24f3a356
