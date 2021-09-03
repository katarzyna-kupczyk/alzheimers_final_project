# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* alzheimers_final_project/*.py

black:
	@black scripts/* alzheimers_final_project/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr alzheimers_final_project-*.dist-info
	@rm -fr alzheimers_final_project.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#      GET DATA FROM GCP
# ----------------------------------
# project id - replace with your GCP project id
PROJECT_ID=winged-axon-319615

# bucket name - replace with your GCP bucket name
BUCKET_NAME=alzheimers-project-699

# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west1

BUCKET_FOLDER=data

LOCAL_PATH="/Users/katarzynakupczyk/code/katarzyna-kupczyk/alzheimers_final_project/raw_data/AlzheimersDataset"

BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

upload_data:
	@gsutil cp -r ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

JOB_NAME=alzheimers_training_model_7_$(shell date + '%Y%m%d_%H%M%S')
BUCKET_TRAINING_FOLDER='trainings'
PACKAGE_NAME=alzheimers_final_project
FILENAME=trainer
PYTHON_VERSION=3.7
RUNTIME_VERSION=2.3

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs \
		--scale-tier CUSTOM \
    --master-machine-type n1-standard-8
