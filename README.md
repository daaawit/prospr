# Important

This repo is a **fork** of [mil-ad/prospr](https://github.com/mil-ad/prospr) that I modified to my needs. **The actual implementation is not mine**, but the original author's. The corresponding paper to the method can be found on ArXiV: [Alizadeh et al. (2022)](https://arxiv.org/abs/2202.08132).

## Changes I made 

For my master's thesis , I rewrote the ProsPr method in a more efficient way that makes use of the prune function that is built in to PyTorch. I'm using the original code as a sanity-checking method, to compare my results at the original results and see whether Im implementation works. To be able to do this, I applied the following changes to this repo: 

* Moved from `python=3.8` to `python=3.10.8`
* Moved from `pytorch=1.9.1` to `pytorch=1.13.1`
  * Required minor changes to different `for` loops, where instance must be checked to avoid errors 
* Added some docstrings for functions I found confusing at first
* Removed anything related to structured pruning as it is irrelevant for my use case
* Removed anything after computing the pruning masks (i.e. application and training), as I don't need the overhead for checking the pruning method (which is done at initialization)
* For my project, I will only require the files from the prospr folder, but I added things to check if the method works with my data/model
  * Use `timm` to load the models instead of the custom-built models by the authors
  * Use the dataloader from my thesis instead of the author's
    * Since I'm CPU bound (for now), I only work with CIFAR10 data here
  * Removed CLI functionality, added main instead



  