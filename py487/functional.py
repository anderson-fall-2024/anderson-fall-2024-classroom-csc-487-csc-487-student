import torch

def elementwise_addition(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Args:
        A: A tensor of any shape.
        B: A tensor of the same shape as A.

    Returns:
        A tensor representing the elementwise sum of A and B.
    """
    # Your solution here
    pass

def concatenate_tensors(A: torch.Tensor, B: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Args:
        A: A tensor of any shape.
        B: A tensor that can be concatenated with A along the specified dimension.
        dim: The dimension along which the concatenation should happen.

    Returns:
        A tensor that is the concatenation of A and B along the given dimension.
    """
    # Your solution here
    pass

def reshape_tensor(A: torch.Tensor, new_shape: tuple) -> torch.Tensor:
    """
    Args:
        A: A tensor of any shape.
        new_shape: A tuple representing the desired shape.

    Returns:
        A tensor reshaped to the specified new shape.
    """
    # Your solution here
    pass

def sum_along_dim(A: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Args:
        A: A 2D tensor.
        dim: The dimension along which the sum will be computed.

    Returns:
        A tensor representing the sum along the given dimension.
    """
    # Your solution here
    pass

def normalize_tensor(A: torch.Tensor) -> torch.Tensor:
    """
    Args:
        A: A tensor of any shape.

    Returns:
        A tensor with values normalized to the range [0, 1].
    """
    # Your solution here
    pass


def matrix_multiply(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Implement this function without using torch.matmul(A, B)

    Args:
        A: A 2D tensor of shape (m, n).
        B: A 2D tensor of shape (n, p).

    Returns:
        A 2D tensor resulting from the matrix multiplication of A and B.
    """
    # Your solution here    
    pass