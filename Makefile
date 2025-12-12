setup_lightning:
	export UV_SYSTEM_PYTHON=1 && uv pip install -e . --no-build-isolation

install_flash_attn3:
	@if [ ! -d "flash-attention" ]; then \
		git clone https://github.com/Dao-AILab/flash-attention.git; \
	fi
	cd flash-attention/hopper && python setup.py install

test:
	python -m pytest tests/ -v

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache */.pytest_cache
	rm -rf *.egg-info
	rm -rf build/ dist/
	rm -rf .ruff_cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

.PHONY: setup_lightning install_flash_attn3 test clean