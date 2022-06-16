sync_landmarks_model = s3cmd get --recursive --skip-existing s3://wave-stylegan2/models/shape_predictor_68_face_landmarks.dat ./models/
sync_attr_models = s3cmd get --recursive --skip-existing s3://wave-stylegan2/models/stylegan2_attributes/ ./models/stylegan2_attributes/

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
	$(sync_landmarks_model)
	$(sync_attr_models)

poetry: ## Install dependencies
	poetry install -vvv

purge: ## Purge previous build
	rm -rf build dist img_styler.egg-info

clean: purge ## Clean. Remove virtual environment and user data
	rm -rf app-data
	rm -rf var
	rm -rf .venv poetry.lock .pytest_cache h2o_wave.state
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
	./.venv/bin/wave run img_styler/ui/app.py

help: ## List all make tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
