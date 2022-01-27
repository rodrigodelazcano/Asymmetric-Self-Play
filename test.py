import torch
import torch.nn.functional as F
from torch import embedding_bag, nn


inputs = torch.randn(10, 2)


aggregation_layer = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=1)
embeddign_layer = nn.Linear(2,4,bias=False)
layer_norm = nn.LayerNorm(4)

print(inputs)
x = embeddign_layer(inputs)
print(x)
print(x.size())
x = layer_norm(x)

y = torch.cat((x,x),-1)
print(x)
print(y)

batch_size = y.size()[0]

y = y.view(-1,2,4)
print('AFTER VIEW', y)

z = aggregation_layer(y)

print(z)
print(z.size())
z = torch.squeeze(z)
print(z)
print(z.size())