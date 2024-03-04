cant_lines ?= 1000

.PHONY: dev
dev:
	streamlit run src/app.py

.PHONY: test
test:
	python src/main.py


.PHONY: build
build:
	python src/main.py $(cant_lines)

.PHONY: models
models:
	python -m spacy download en_core_web_sm
	python -m spacy download es_core_news_sm