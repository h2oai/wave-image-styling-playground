sync_landmarks_model = s3cmd get --recursive --skip-existing s3://ai.h2o.wave-image-styler/public/models/shape_predictor_68_face_landmarks.dat ./models/
sync_attr_models = s3cmd get --recursive --skip-existing s3://ai.h2o.wave-image-styler/public/models/stylegan2_attributes/ ./models/stylegan2_attributes/
sync_stgan_nada_models = s3cmd get --recursive --skip-existing s3://ai.h2o.wave-image-styler/public/models/stylegan_nada/ ./models/stylegan_nada/
sync_gfpgan_models = s3cmd get --recursive --skip-existing s3://ai.h2o.wave-image-styler/public/models/gfpgan/ ./models/gfpgan/
download_ffhq_model = wget -P  ./models/ https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
download_sd_model  = s3cmd get --recursive --skip-existing s3://h2o-model-gym/models/stable-diffusion-v1-4/ ./models/stable_diffusion_v1_4/

.PHONY: download_models

all: download_models ## Build app
	h2o bundle

setup: download_models ## Install dependencies
	mkdir -p var/lib/tmp/jobs
	mkdir -p var/lib/tmp/jobs/output

	python3 -m venv .venv
	./.venv/bin/python -m pip install --upgrade pip
	./.venv/bin/python -m pip install -r requirements_dev.txt

download_models:
	mkdir -p models
	mkdir -p models/stylegan2_attributes
	mkdir -p models/stylegan_nada
	mkdir -p models/gfpgan/
	mkdir -p models/stable_diffusion_v1_4
	$(download_ffhq_model)
	$(sync_landmarks_model)
	$(sync_attr_models)
	$(sync_stgan_nada_models)
	$(sync_gfpgan_models)
	$(download_sd_model)

poetry: ## Install dependencies
	poetry install -vvv

purge: ## Purge previous build
	rm -rf build dist img_styler.egg-info

clean: purge ## Clean. Remove virtual environment and user data
	rm -rf app-data
	rm -rf var
	rm -rf .venv .pytest_cache h2o_wave.state
	find . -type d -name "__pycache__" -exec rm -rf \;

reset: ## Delete the cached user data and start fresh
	rm -rf app-data
	rm -rf var
	rm -rf *.wave
	rm -rf .pytest_cache h2o_wave.state
	find . -type d -name "__pycache__" -exec rm -rf \;

run: ## Run the app with no reload
	./.venv/bin/wave run --no-reload img_styler/ui/app.py

dev: ## Run the app with active reload
	export H2O_WAVE_NO_LOG=1; export H2O_WAVE_MAX_REQUEST_SIZE=20M; ./.venv/bin/wave run img_styler/ui/app.py

help: ## List all make tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
