# MLMCBO description

This file is for internal usage. Remove this file for publication.

This file describe how this package works with work flow and what each file contains. 

## Files
"./acquisition_functions":
- mc_one_step_lookahead.py (legacy - will be deleted later)

    Onestep lookahead EI
    Antithetic coupling EI

- mlmc_inc_functions.py
    
    One-step lookahead MLMC increment

    Custom multistep lookahead acquisition function
    
    One-step lookahead antithetic coupling EI

    Log EI for MLMC (underworking)

    Two-step lookahead MLMC increment (undertesting)
    
    Two-step lookahead coupling EI (undertesting)

"./utils":
- ```model_fit.py```

    Definition of the surrogate model - Gaussian process

- ```objectiveFunctions.py```

    Self-defined function and functions corresponding to Frazier's paper that different from the standard implementation

- ```optimize_mlmc.py```

    Main function for MLMC
    - iterating from l=0 to l=L if using point matching or forward matching
    - iterating form l=L to l=0 if using backward matching

"./tutorials":
- ```tests.py``` (legacy - will be deleted later)

    Include a quick guidance on how to run MC and MLMC BO with one iteration

- ```runBO.py```
 
    BO routine - parameter ML allows choosing to run BO with MC or MLMC

- ```testWholeBO.py```
    
    Calling MC and MLMC BO for comparison and visualise the results

## Work flows

One-step lookahead EI:
```testWholeBO.py``` &rarr; ```runBO``` &rarr; ```GPmodel```;```qEIMLMCOneStep```;```optimize_mlmc``` 

- In ```testWholeBO.py```, 
  - commenting out different synthetic function for different objective
  - setting q = [1, 2] for 1EI + 2EI, q = [2, 2] for 2EI + 2EI
  - changing match_mode for different matching strategy
  - changing kernel for different kernels of GP

Two-step lookahead EI:
```testTwoEI.py``` &rarr; ```runBO``` &rarr; ```GPmodel```;```qEIMLMCTwoStep```;```optimize_mlmc_two``` 

- In ```testTwoEI.py```, 
  - commenting out different synthetic function for different objective
  - setting q = [1, 1, 1] for EI + EI + EI (q > 1 is not implemented - future work)
  - changing match_mode for different matching strategy
  - changing kernel for different kernels of GP