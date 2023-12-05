# NDDL-predictive-coding
Team Parva:

Intro: 
> Predictive coding


![image info](./predcode_scheme-02.jpg)



## Todo list:
- Optimize code for batch training
- (Individually) Write out project proposal plans
- ...  

## Possible extensions:
- Supervised learning
- Improve the quality of the latent representations learnt by the model (Changing architecture, Connecting the two modalities at earlier layers, lateral inhibition, precision weighting, etc)
- Extend the model (Other modalities, temporal dynamics, mini-batches, etc)
- Improve biological plausibility of the model (More realistic architectures, Excitatory/Inhibitory neurons, etc)
- Content-addressable memory: denoising and occlusion (Salvatori et al., 2023, p. 7; Papadimitriou et al., 2020)
- Learning rate decay mechanism (e.g. RMSprop). Also: decay of update ratios between representation and weights. 
- Storage capacity auto-associative memory (i.e. calc max storage capacity; Salvatori et al., 2021)
- Lateral processing between different modalities
- Non-symmetrical weights (as in neural generative coding)
- Implement probability distribution (Rao & Ballard -> FEP)


## Multi-Model Datasets:
- https://ai.meta.com/blog/ai-self-supervised-learning-data2vec/
- https://github.com/drmuskangarg/Multimodal-datasets
