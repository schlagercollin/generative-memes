import torch
import torchvision
import torchvision.transforms as T 
from PIL import Image

def tensor_to_image(output: torch.Tensor, ncol: int=4, padding: int=2) -> Image:
    """Convert the tensor-based output from a Generator into a PIL image.
    
    Note: use np.asarray(img) to convert to a numpy array.

    Args:
        output (torch.Tensor): output from Generator model
        ncol (int, optional): number of columns. Defaults to 4.
        padding (int, optional): padding around each image. Defaults to 2.

    Returns:
        Image: PIL image object
    """
    out = torchvision.utils.make_grid(output, nrow=ncol, normalize=True, padding=padding)
    return T.ToPILImage()(out)

def interpolate(a: torch.Tensor, b: torch.Tensor, num_steps: int=64) -> torch.Tensor:
    """Linearly interpolate between tensors A and B with num_steps steps.

    Args:
        a (torch.Tensor): Starting point.
        b (torch.Tensor): Ending point.
        num_points (int, optional): Number of intermediate tensors to create. Defaults to 64.

    Returns:
        torch.Tensor: (num_steps, **a.shape) sized tensor.
    """
    
    # TODO: add different types of interpolation
    
    if a.shape != b.shape:
        raise ValueError(f"a and b need the same shape. got {a.shape} and {b.shape}.")
    
    return torch.stack([torch.lerp(a, b, weight) for weight in torch.linspace(0, 1, num_steps)])