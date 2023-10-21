# VG-MLM: Visually Grounded Masked Language Model

Code for the paper *[Unsupervised Scene Graph Induction from Natural Language Supervision](./paper.pdf)* by [Yanpeng Zhao]() and [Ivan Titov](http://ivan-titov.org).

<p align="center"><img src="https://drive.google.com/uc?id=1DdPMr7jUwS5TLgjd--WqMR75ceGHCCue" alt="VG-MLM is trained via multimodal masked language modeling" width=75%/></p>

## Data

Currently we consider only artificial datasets such as [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) and [AbstractScene](http://optimus.cc.gatech.edu/clipart/). We focus on CLEVR but may also discuss the pre-processing of AbstractScene.

### Generate Bounding Boxes for CLEVR

We use [CELVR v1.0](https://cs.stanford.edu/people/jcjohns/clevr/) and rely on [this](https://github.com/ccvl/clevr-refplus-dataset-gen) repo to generate bounding boxes for the CLEVR images.

For AbstractScene images, bounding boxes can be automatically inferred (see [this](https://github.com/zhaoyanpeng/sgi/blob/c74e0bece250382de416685c55d3c8143a01b01c/sgi/data/scene_function.py#L136) script).

### Pre-encode Images

Check out the running script `bash/run_obj_enc.sh`. 

### Generate Captions

Check out [this](https://github.com/zhaoyanpeng/clevr-tv) repo.

## Learning

Learning with *symbolic* object representations: `bash/run_clevr_sym.sh`.

Learning with *visual* object representations: `bash/run_clevr_vis.sh`.

Learning with additional causal language modeling (CLM): `bash/run_clevr_clm.sh`.

## Inference

Currently we do not have a specific script for inference since evaluation (i.e., inference) is automatically performed after every training epoch. If you want to create an inference script, you may want to check out `bash/run_clevr_eval.sh` and the associated inference [function](https://github.com/zhaoyanpeng/sgi/blob/c74e0bece250382de416685c55d3c8143a01b01c/sgi/monitor/monitor.py#L242).

## Dependencies

```shell
git clone --branch beta https://github.com/zhaoyanpeng/sgi.git
cd sgi
virtualenv -p python3.8 ./pyenv/oops
source ./pyenv/oops/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install -r requirements.txt
```

## Citation

```bibtex
@misc{VC-MLM,
  author = {Yanpeng Zhao and Ivan Titov},
  title = {Unsupervised Scene Graph Induction from Natural Language Supervision},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/zhaoyanpeng/sgi}},
}
```

## License
MIT
