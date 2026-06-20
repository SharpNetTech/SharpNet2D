<div align="center">
<h1>SharpNet: Enhancing MLPs to Represent Functions with Controlled Non&#8288;-&#8288;differentiability</h1>

[ACM DL](https://doi.org/10.1145/3811330) | [Homepage](https://sharpnettech.github.io) | <b>SharpNet2D</b> | Coming soon...

ACM Transactions on Graphics (SIGGRAPH 2026)

<p><span><b>Hanting Niu</b><sup>1,2,*</sup></span> · <span><b>Junkai Deng</b><sup>3,*</sup></span> · <span><b>Fei Hou</b><sup>1,2</sup></span> · <span><b>Wencheng Wang</b><sup>1,2</sup></span> · <span><b>Ying He</b><sup>3</sup></span></p>
<p><sup>1</sup> Institute of Software, Chinese Academy of Sciences<br>
<sup>2</sup> University of Chinese Academy of Sciences<br>
<sup>3</sup> Nanyang Technological University</p>
<p><sup>*</sup> Equal contributions</p>
</div>

## SharpNet2D ##

This is the official code release for paper "SharpNet: Enhancing MLPs to Represent Functions with Controlled Non-differentiability", the 2D experiment part (Section 4).

## What does this repo do? ##
This repo should be able to reproduce the following experiments:
| | Geodesic<br>(Section 4.1) | Medial axis<br>(Section 4.2) | Belhe<br>(Section 4.3) |
|--|:--:|:----:|:----:|
| Raw MLP | ✓ | - | ✓ |
| InstantNGP | ✓ | - | ✓ |
| SharpNet w/ ReLU | ✓ | - | ✓ |
| SharpNet w/ Softplus (Ours) | ✓ | ✓ | ✓ |
| Belhe et al | - | - | ✗ |
| Liu et al | - | - | ✗ |

Note: The experiment in Section 4.3 is conveniently named "Belhe" because the feature edges are taken directly from Belhe et al. It should not be confused with the actual method.

## Environment setup ##

### Docker ###
We provide a dockerfile that will set up a consistent runtime environment. The dockerfile will set up necessary dependencies for all experiments including InstantNGP (`tiny-cuda-nn`).

```bash
docker -t sharpnet2d:latest .
```

Boot the image as
```bash
docker run --rm -it --gpus=all --ipc=host -v /path/to/local:/path/in/container sharpnet2d:latest bash
```

### Pip ###
We provide a pip specification `requirements.txt`. It installs all necessary dependencies for SharpNet. **It does not install InstantNGP (`tiny-cuda-nn`).** You can always uncomment the last line of the file to install it.

```bash
pip install -r requirements.txt
```

### Conda ###
We also provide a conda environment specification. It does not do much besides asking for a Python at least 3.10 and older than 3.13, then install the dependencies via pip.

```bash
conda create -n sharpnet2d -f environment.yml
```

### Notes ###
This repo uses two beta features of PyTorch as of Jan 2026.
* `torch.func`: we use the `vmap` method;
* `torch.sparse`: we use this to support mollifier acceleration.

The experiments are done in PyTorch 2.8 and we don't recommend using other major versions because beta features may be subject to changes between versions. This also means that the support for these features across platforms may be limited. Notably, **you might not be able to run our code with mollifier on Mac MPS.** Please help PyTorch developers stabilize these valuable features.

InstantNGP (`tiny-cuda-nn`) is optional. The code will load normally if it is not installed (we handle the exception) and only a warning message will be emitted. If you are not running any InstantNGP experiments you can disregard it. The code will catch fire if you are actually running InstantNGP experiments without `tiny-cuda-nn`.

## Run the experiments ##
Different experiments have different experiment runners. These runners are stored in the `runner` folder.
* `runner/geodesic.py` for the Geodesic experiment;
* `runner/medial.py` for the Medial axis experiment;
* `runner/belhe.py` for the Belhe experiment.

The corresponding methods (configurations) are organized in the `confs` folder.
* `confs/geodesic` for the Geodesic experiment;
* `confs/medial/rectangle` for the Medial axis experiment;
* `confs/belhe` for the Belhe experiment.

Inside each configuration folder, the following configurations may present:
* `pe.conf` for Raw MLP (PE is short for positional encoding);
* `ingp.conf` for InstantNGP;
* `sharp_pe_relu.conf` for SharpNet with ReLU activation;
* `sharp_pe.conf` for SharpNet with Softplus activation.

The experiments are run in a similar manner to NeuS.
```bash
python /path/to/runner.py --conf /path/to/configuration.conf --mode train
```

## We encourage hacking ##
We are currently working to compile a developer's notes that explains the structure and algorithm of various parts of the code. It will ease your mental burden of understanding our code, but it is a burden to us. Please be patient.

## Citation ##
If you find our work useful, please cite SharpNet.
```bibtex
@article{niu2026sharpnet,
    author = {Niu, Hanting and Deng, Junkai and Hou, Fei and Wang, Wencheng and He, Ying},
    title = {{SharpNet}: Enhancing {MLP}s to Represent Functions with Controlled Non-differentiability},
    year = {2026},
    issue_date = {July 2026},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {45},
    number = {4},
    issn = {0730-0301},
    url = {https://doi.org/10.1145/3811330},
    doi = {10.1145/3811330},
    journal = {ACM Transactions on Graphics},
    month = jul,
    articleno = {113},
    numpages = {19},
    keywords = {MLP, Sharp features, Poisson's equation, Jump Neumann boundary condition, Green's function, CAD},
}
```
