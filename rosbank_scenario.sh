#!/bin/bash

# Set the range of values for frac parameter
for frac in $(seq 0.1 0.1 1.0); do

    # Set the list of values for hidden_size parameter
    for hidden_size in 64 128 256 512; do

        # Run the Python file with the current parameters
        python Rosbank.py --frac $frac --hidden_size $hidden_size
    
    done
done