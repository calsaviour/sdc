1. in_j = sumk(w_k,j * a_k)				// Forward pass starts. Compute weighed input to all hidden units
2. a_j = sigmoid(in_j)					// Compute outputs at all hidden units
3. in_i = sumj(w_j,i * a_j)				// Compute weighed inputs to all output units
4. a_i = sigmoid(in_i)					// Compute outputs at all output units









5. del_i = a_i * (1 - a_i) * (y_i - a_i)		// Backward pass starts.  Compute "modified error" at output units
6. del_j = a_j * (1 - a_j) * sumi(w_j,i * del_i)	// Compute "modified error" at all hidden units
7. w_j,i = w_j,i + (alpha * a_j * del_i)		// update weights between hidden and output units
8. w_k,j = w_k,j + (alpha * a_k * del_j)		// update weights between input and hidden units