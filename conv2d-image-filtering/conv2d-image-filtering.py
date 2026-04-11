import torch

def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    # Write code here
    image = torch.tensor(image, dtype=torch.float32)
    kernel = torch.tensor(kernel, dtype=torch.float32)
    image = image.unsqueeze(0).unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    print(f"Before conv2d: {image}, {kernel}")
    image_conv = torch.nn.functional.conv2d(image, kernel, stride=stride, padding=padding)
    print(f"After conv2d: {image_conv}")
    ans = image_conv.squeeze(0).squeeze(0)
    ans = ans.tolist()
    return ans
    