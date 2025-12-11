all: transform

extract:
	@python src/extract.py

transform:
	@python src/transform.py
