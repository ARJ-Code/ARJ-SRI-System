cant ?= 100

.PHONY: dev
dev:
	streamlit run src/app.py

.PHONY: test
test:
	python src/main.py


.PHONY: build
build:
	python src/main.py $(cant)

.PHONY: metrics
metrics:
	python src/metrics.py $(cant)


.PHONY: models
models:
	python -m spacy download en_core_web_sm
	python -m spacy download es_core_news_sm