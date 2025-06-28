.PHONY: deps_table_update modified_only_fixup extra_style_checks quality style fixup fix-copies test test-examples

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := examples scripts src tests utils benchmarks

modified_only_fixup:
	$(eval modified_py_files := $(shell python utils/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		ruff check $(modified_py_files) --fix; \
		ruff format $(modified_py_files);\
	else \
		echo "No library .py files were modified"; \
	fi

# Update src/diffusers/dependency_versions_table.py

deps_table_update:
	@python setup.py deps_table_update

deps_table_check_updated:
	@md5sum src/diffusers/dependency_versions_table.py > md5sum.saved
	@python setup.py deps_table_update
	@md5sum -c --quiet md5sum.saved || (printf "\nError: the version dependency table is outdated.\nPlease run 'make fixup' or 'make style' and commit the changes.\n\n" && exit 1)
	@rm md5sum.saved

# autogenerating code

autogenerate_code: deps_table_update

# Check that the repo is in a good state

repo-consistency:
	python utils/check_dummies.py
	python utils/check_repo.py
	python utils/check_inits.py

# this target runs checks on all files

quality:
	ruff check $(check_dirs) setup.py
	ruff format --check $(check_dirs) setup.py
	doc-builder style src/diffusers docs/source --max_len 119 --check_only
	python utils/check_doc_toc.py

# Format source code automatically and check is there are any problems left that need manual fixing

extra_style_checks:
	python utils/custom_init_isort.py
	python utils/check_doc_toc.py --fix_and_overwrite

# this target runs checks on all files and potentially modifies some of them

style:
	ruff check $(check_dirs) setup.py --fix
	ruff format $(check_dirs) setup.py
	doc-builder style src/diffusers docs/source --max_len 119
	${MAKE} autogenerate_code
	${MAKE} extra_style_checks

# Super fast fix and check target that only works on relevant modified files since the branch was made

fixup: modified_only_fixup extra_style_checks autogenerate_code repo-consistency

# Make marked copies of snippets of codes conform to the original

fix-copies:
	python utils/check_copies.py --fix_and_overwrite
	python utils/check_dummies.py --fix_and_overwrite

# Run tests for the library

test:
	docker run \
    --network host \
    --shm-size 10.24g \
    --privileged \
    --rm \
    -e JAX_HBM_DEFRAGMENTATION_LEVEL=1 \
    -v /data/tmp:/tmp \
    -v /data/huggingface_cache:/root/.cache/huggingface \
    -v /home/greg_greghuang_altostrat_com/diffusers:/root/diffusers \
    us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.11_tpuvm_20250617 \
    bash -c "pip install -q transformers accelerate ftfy imageio opencv-python imageio-ffmpeg jax[tpu] flax && \
      python3 /root/diffusers/src/wan_tx_splash_attn.py"

# Run tests for examples

test-wan:
	docker run \
    --network host \
    --shm-size 10.24g \
    --privileged \
    --rm \
    -e JAX_HBM_DEFRAGMENTATION_LEVEL=1 \
    -v /data/tmp:/tmp \
    -v /data/huggingface_cache:/root/.cache/huggingface \
    -v /home/greg_greghuang_altostrat_com/diffusers:/root/diffusers \
    us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.11_tpuvm_20250617 \
    bash -c "pip install -q transformers accelerate ftfy imageio opencv-python imageio-ffmpeg jax[tpu] flax && \
      python3 /root/diffusers/src/wan_tx.py"


# Release stuff

pre-release:
	python utils/release.py

pre-patch:
	python utils/release.py --patch

post-release:
	python utils/release.py --post_release

post-patch:
	python utils/release.py --post_release --patch
