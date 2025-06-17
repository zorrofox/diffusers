Original readme moved to README_original.md

# install
```
pip install -e .
pip install transformers accelerate

# install torchax
pip install git+https://github.com/pytorch/xla.git@hanq_wan_changes
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jax[tpu]
```

To run:

```
python wan_tx.py
```


## sizes:
### wan 1.3B:
text_encoder 12.537574768066406 G
transformer 2.64891254901886 G
vae 0.23635575734078884 G

### wan 14B
text_encoder 12.537574768066406 G
transformer 26.66874897480011 G
vae 0.23635575734078884 G

