# Dataset Description
The data used in this example is a subset of the dataset used in [MVDA][1].
The dataset comprises 114 observations of about 4000 mRNA expression levels obtained by sequencing technologies.

[1]: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0680-3

# Setup
The example has been developed and tested using the Miniconda distribution of Python 2.7.
See the Miniconda [website][2] for installation instructions.

## Dependencies installation
To run the code and reproduce the results, the python data science stack needs to be installed:

```Bash
conda install mkl numpy scipy pandas sciki-learn matplotlib jupyter
```

NB: while the MKL library is optional, it is highly recommended to improve the performances of numeric procedures

[2]: https://conda.io/miniconda.html
