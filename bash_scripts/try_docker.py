import torch
import torch.utils
import torch.utils.cpp_extension

print(torch.__version__)

print(torch.version.cuda)
print(torch.cuda.is_available())