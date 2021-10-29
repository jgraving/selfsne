Contrastive Noise Embeddings (CNE) for dimensionality reduction and clustering
============
<p align="center">
<img src="https://github.com/jgraving/cne/blob/master/assets/header.jpeg" max-height:256px>
</p>

CNE is a probabilistic self-supervised deep learning model for compressing high-dimensional data to a low-dimensional embedding. CNE is a general-purpose algorithm that works with multiple types of data including images, sequences, and tabular data. It uses the InfoNCE objective, a variational bound on mutual information based on noise contrastive estimation (NCE), to preserve local structure in the compressed latent space. CNE also simultaneously learns a cluster distribution (a prior over the latent embedding) during optimization, and overlapping clusters are automatically combined by optimizing a variational upper bound on entropy, so the number of clusters does not have to be specified manually — provided the number of initial clusters is large enough. CNE produces embeddings with similar quality to existing dimensionality reduction methods; can detect outliers; scales to large, out-of-core datasets; and can easily add new data to an existing embedding/clustering.

This is an alpha release currently undergoing development. Features may change without notice. Use at your own risk.


References
------------
If you use CNE for your research please cite [version 1 of our preprint](https://doi.org/10.1101/2020.07.17.207993) (an updated version is forthcoming):

    @article{graving2020vae,
    	title={VAE-SNE: a deep generative model for simultaneous dimensionality reduction and clustering},
    	author={Graving, Jacob M and Couzin, Iain D},
    	journal={BioRxiv},
    	year={2020},
    	publisher={Cold Spring Harbor Laboratory}
    }



License
------------
Released under a Apache 2.0 License. See [LICENSE](https://github.com/jgraving/cne/blob/master/LICENSE) for details.
