Self-Supervised Noise Embeddings (Self-SNE) for dimensionality reduction and clustering
============
__This is an alpha release currently undergoing development__. Examples and documentation will be added upon release of the accompanying publication. 
Not all features have been validated and may change without notice. Use at your own risk.

<p align="center">
<img src="https://github.com/jgraving/selfsne/blob/main/assets/header.png" max-height:256px>
</p>

Self-SNE is a probabilistic family of self-supervised deep learning models for compressing high-dimensional data to a low-dimensional embedding. It is a general-purpose algorithm that works with multiple types of data including images, sequences, and tabular data. It uses self-supervised objectives to preserve structure in the compressed latent space. Self-SNE can also (optionally) simultaneously learn a cluster distribution (a prior over the latent embedding) during optimization.


References
------------
If you use Self-SNE for your research please cite [ 1 of our preprint](https://doi.org/10.1101/2020.07.17.207993) (an updated version is forthcoming):

    @article{graving2020vae,
    	title={VAE-SNE: a deep generative model for simultaneous dimensionality reduction and clustering},
    	author={Graving, Jacob M and Couzin, Iain D},
    	journal={BioRxiv},
    	year={2020},
    	publisher={Cold Spring Harbor Laboratory}
    }



Licenseversion
------------
Released under a Apache 2.0 License. See [LICENSE](https://github.com/jgraving/cne/blob/main/LICENSE) for details.
