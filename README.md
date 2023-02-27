# ISMKL -- In-Situ Multiple Kernel Learning
ISMKL implementation based on the paper 'A Multiple Kernel Machine with Incremental Learning using Sparse Representation' by Ali Pezeshki, Mahmood R. Azimi-Sadjadi, and Christopher Robbiano.

An example of how to run the code is in the `scripts/learn_yale_data.ipynb` notebook.

Quick notes:
- Data is expected to be in numpy matrices, with each sample being a new column, i.e., 50 samples of 1024 features would be in a matrix of `1024x50`
- 

TODO:
- Will maybe someday port this into tensorflow for faster operations
- Cleanup the fixme's
- Change the criteria for deciding if a sample is correctly estimated
  - currently looks at a predetermined distance between the estimated likelihood of the top two class labels
  - could just look to see if it actually estimated correctly
- I'm sure theres more