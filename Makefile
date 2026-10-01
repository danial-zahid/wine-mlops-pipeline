.PHONY: install lint test train clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	flake8 src/ tests/ --max-line-length=100

test:
	pytest tests/ -v

train:
	python src/train.py

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"