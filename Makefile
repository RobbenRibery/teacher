setup_lightning:
	export UV_SYSTEM_PYTHON=1 && uv pip install -e . --no-build-isolation

install_flash_attn3:
	@if [ ! -d "flash-attention" ]; then \
		git clone https://github.com/Dao-AILab/flash-attention.git; \
	fi
	cd flash-attention/hopper && python setup.py install