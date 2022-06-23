# gan4arome_public

*Public version of gan4arome repo.*

This repo is only intended to keep up-to-date accessible versions of the code.
It is essentially work in progress that should be completed on the long run.
It should therefore not be expected any type of replicability.

Distributed under CC-BY-SA, free for personal use and modification of any kind.

# In short :

*experimental* : contains unfinished/untested features related to architectures, data preprocessing, training algorithms

*importance sampling* : contains scripts to resample data according to their "importance" (amount of meteorological information they contain)

*gan_horovod* : contains up-to-date GAN training logics, networks architectures and  data pipelining, interfacing horovod API.

*gan_std* : is the original library (not up-to-date) containing basic training logics and no interface for multi-GPU.

*metrics4arome* : contains the implementations of the many metrics used to compare GAN and PEARO outputs, together with short snippets to test them. Includes spectral analysis, Wasserstein distances implementations and scattering transform analysis.

*score_crawl* : contains automation scripts to apply metrics on diverse sources of data using different modes (parallelization, GPU/CPU).


# Requirements :
torch, numpy, horovod, kymatio
