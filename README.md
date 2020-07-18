VAE-SNE: a deep generative model for dimensionality reduction and clustering
============
<p align="center">
<img src="https://github.com/jgraving/vaesne/blob/master/assets/vaesne_figure.jpeg" max-height:256px>
</p>

VAE-SNE is a deep generative model for both dimensionality reduction and clustering. VAE-SNE is a variational autoencoder (VAE) optimized using the ELBO objective regularized with the stochastic neighbor embedding (t-SNE/SNE) objective to improve local structure preservation. The model simultaneously learns a Gaussian mixture cluster distribution during optimization, and overlapping mixture components are then combined using a sparse watershed procedure, so the number of clusters does not have to be specified manually provided the number of Gaussian mixture components is large enough. VAE-SNE performs similarly to existing dimensionality reduction methods; can detect outliers; scales to large, out-of-core datasets; and can easily add new data to an existing clustering/embedding.

The code and documentation for VAE-SNE is coming soon. For now you can read more about it in our preprint:
[Graving, Jacob M., and Couzin, Iain D. 2020. VAE-SNE: a deep generative model for simultaneous dimensionality reduction and clustering. https://doi.org/10.1101/2020.07.17.207993](https://doi.org/10.1101/2020.07.17.207993).


License
------------
Released under a Apache 2.0 License. See [LICENSE](https://github.com/jgraving/vaesne/blob/master/LICENSE) for details.
