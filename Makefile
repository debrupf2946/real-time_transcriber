GIT_ROOT := $(shell git rev-parse --show-toplevel)
INFRA_ROOT := $(GIT_ROOT)/infra
VERSION ?= local
RELEASE_VERSION ?= v1
TF_INFRA_ROOT := $(INFRA_ROOT)
ECR_REPO_BASE_URL = 975226449092.dkr.ecr.us-east-1.amazonaws.com
WHISPER_RAY_SERVICE_REPO_URL = $(ECR_REPO_BASE_URL)/whisper-ray-service
EKS_CLUSTER_NAME := kuberay-cluster
REGION = us-east-1
TARGET_ZIP=whisper-ray-service.zip
SOURCE_FILES=.

ecr-login:
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(ECR_REPO_BASE_URL)

set-jarvis-kubectl-context:
	aws eks update-kubeconfig --region $(REGION) --name $(EKS_CLUSTER_NAME)

build-ray-service:
	docker build --platform linux/amd64 --no-cache -t whisper-ray-service:$(VERSION) .

push-docker-whisper-ray-service:
	docker tag whisper-ray-service:$(VERSION) $(WHISPER_RAY_SERVICE_REPO_URL):$(RELEASE_VERSION)
	docker push $(WHISPER_RAY_SERVICE_REPO_URL):$(RELEASE_VERSION)

build-push-ray-service: build-ray-service push-docker-whisper-ray-service

deploy-ray-service:
	kubectl apply -f $(GIT_ROOT)/Whisper-RayService.yaml

undeploy-ray-service:
	kubectl delete -f $(GIT_ROOT)/Whisper-RayService.yaml

# create-zip:
# 	zip -r whisper-ray-service.zip . -x "infra/*" ".git/*" 
# 	old_file="whisper-ray-service.zip"
# 	new_file="$(md5sum "$old_file" | awk '{print $1}').zip"
# 	mv "$old_file" "$new_file"
# 	echo "Created zip file: $new_file"


# Rule to create a ZIP file
create_zip:
	@echo "Creating ZIP file: $(TARGET_ZIP)"
	@zip -r "$(TARGET_ZIP)" . -x "infra/*" ".git/*" 
 
# Rule to rename the ZIP file using SHA256 hash
rename_zip: create_zip
	@hash=$$(sha256sum "$(TARGET_ZIP)" | awk '{print $$1}'); \
	mv "$(TARGET_ZIP)" "$$hash.zip"; \
	echo "Renamed $(TARGET_ZIP) to $$hash.zip"


# upload-to-s3:

# 	aws s3 cp my_archive.zip s3://your-bucket-name/ --acl public-read
