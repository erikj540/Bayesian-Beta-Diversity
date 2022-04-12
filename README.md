# Installation and requirements

The code for our Bayesian repertoire overlap paper is available as a python package called `bro`. To install the `bro` package, the `pip` package manager needs to be installed. Once `pip` is installed, from the parent directory that contains this README (i.e., the `Bayesian-Beta-Diversity` directory), you can install the `bro` package via
```
pip install -e ./
```

To verify that the package installed, run `test.py` via
```
python test.py
```

If there are no errors, then you should be good to go.

# Getting started

Given count data data from two sets, A and B, of size Ra and Rb, respectively, with s elements in common, the method described in our paper computes the joint posterior distribution of (s, Ra, Rb) given the count data and priors on Ra and Rb. 

There is a helper function `calculate_joint_pdf_from_count_data` in the class `DistributionComputations` that computes the joint distribution of (s, Ra, Rb) via
```
comp_engine = DistributionComputations()
joint_pdf = comp_engine.calculate_joint_pdf_from_count_data(count_data, Ra_prior, Rb_prior)
```

To use `calculate_joint_pdf_from_count_data`, the assumption are
* the `count_data` object has attributes `na`, `nb`, `nab`, `count_dataA`, and `count_dataB` where
    - `na` is the number of distinct items drawn from set A
    - `nb` is the number of distinct items drawn from set B 
    - `nab` is the number of distinct items drawn from both set A and set B
    - `count_dataA` are all the items drawn from set A where items can be drawn more than once. `count_dataA` should be a list with ma items where ma is the total number of draws from set A. The assumption is that the items in set A have been indexed so that `count_dataA` is list of integers. The same assumptions apply to `count_dataB`.
* `Ra_prior` is an Nx2 numpy array with the first column being the values Ra can take and the second column being the probability of those values (i.e., p(Ra)). And the same is true for `Rb_prior`. 

# Example
For a more detailed example that includes generating synthetic count data and computing marginal distributions and credible intervals from the joint PDF, check out the `example_notebook.ipynb` Jupyter notebook in this repo. 

# Figures and data
We have included the data used to make the figures in our paper in the `data` folder. The Jupyter notebook `paper_figures.ipynb` recreates the figures in our paper using the included data.