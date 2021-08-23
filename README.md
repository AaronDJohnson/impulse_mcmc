# impulse
 Rewrite of PTMCMCSampler focusing on modularity

## Currently working:
  * Metropolis Hastings sampler for one core
  * Modular jump proposals and ability to mix proposals
  * HDF5 saving after sampler is finished

## Currently under development:
  * Parallel Tempering MCMC
  * Using Schwimmbad to swap between MPI and multiprocessing
  * Resume function
  * Convergence tests

## Future feature list:
  * Master-slave paradigm for scalable parallel tempering to arbitrary numbers of CPUs
  * Convergence tests for burn-in and stop conditions (Geweke, Gelman-Rubin r-hat, etc.)
