.PHONY: help setup preprocess train-autoencoder train-ttl train-duration evaluate inference clean install

help:
	@echo "SupertonicTTS Hindi TTS Training - Available Targets"
	@echo "===================================================="
	@echo "make setup              - Install dependencies and setup environment"
	@echo "make install            - Install package in development mode"
	@echo "make preprocess         - Preprocess audio and text data"
	@echo "make train-autoencoder  - Train speech autoencoder"
	@echo "make train-ttl          - Train text-to-latent model"
	@echo "make train-duration     - Train duration predictor"
	@echo "make evaluate           - Evaluate all models"
	@echo "make inference          - Run inference on test samples"
	@echo "make clean              - Clean up generated files and caches"
	@echo ""

setup: install preprocess
	@echo "Setup complete! You can now start training."

install:
	pip install -e .
	@echo "Installation complete!"

preprocess:
	python -m supertonic_hindi_tts.data.preprocess \
		--config configs/base_config.yaml \
		--hindi-config configs/hindi_data_config.yaml \
		--output-dir ./data/processed
	@echo "Data preprocessing complete!"

train-autoencoder:
	python -m supertonic_hindi_tts.training.train_autoencoder \
		--config configs/base_config.yaml \
		--hindi-config configs/hindi_data_config.yaml \
		--checkpoint-dir ./outputs/autoencoder \
		--log-dir ./outputs/logs/autoencoder
	@echo "Autoencoder training complete!"

train-ttl:
	python -m supertonic_hindi_tts.training.train_text_to_latent \
		--config configs/base_config.yaml \
		--hindi-config configs/hindi_data_config.yaml \
		--autoencoder-ckpt ./outputs/autoencoder/latest.ckpt \
		--checkpoint-dir ./outputs/text_to_latent \
		--log-dir ./outputs/logs/text_to_latent
	@echo "Text-to-latent training complete!"

train-duration:
	python -m supertonic_hindi_tts.training.train_duration_predictor \
		--config configs/base_config.yaml \
		--hindi-config configs/hindi_data_config.yaml \
		--autoencoder-ckpt ./outputs/autoencoder/latest.ckpt \
		--checkpoint-dir ./outputs/duration_predictor \
		--log-dir ./outputs/logs/duration_predictor
	@echo "Duration predictor training complete!"

evaluate:
	python -m supertonic_hindi_tts.evaluation.evaluate_all \
		--config configs/base_config.yaml \
		--hindi-config configs/hindi_data_config.yaml \
		--autoencoder-ckpt ./outputs/autoencoder/latest.ckpt \
		--ttl-ckpt ./outputs/text_to_latent/latest.ckpt \
		--duration-ckpt ./outputs/duration_predictor/latest.ckpt \
		--output-dir ./outputs/evaluation
	@echo "Evaluation complete!"

inference:
	python -m supertonic_hindi_tts.inference \
		--config configs/base_config.yaml \
		--hindi-config configs/hindi_data_config.yaml \
		--autoencoder-ckpt ./outputs/autoencoder/latest.ckpt \
		--ttl-ckpt ./outputs/text_to_latent/latest.ckpt \
		--duration-ckpt ./outputs/duration_predictor/latest.ckpt \
		--output-dir ./outputs/inference
	@echo "Inference complete!"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/
	@echo "Clean complete!"
