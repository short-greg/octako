# Tako

Design more flexible learning machines and 

## Diving In

Tako consists of two major components.

* **net**: Classes for building and querying a network 
* **learn**: Classes for training a network

With net you can generate a network dynamically and also iterate over the network.

```
class T(Tako):

  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(2, 2)

  def forward_iter(self, in_: In) -> typing.Iterator:
    
    layer = in_.to(self.linear, name=self.x)
    yield layer
    y = layer.to(torch.sigmoid, name=self.y)
    yield y


tako = T()

```

You can then pass in an input to the Tako as you normally would with an Torch nn.Module.

```
print(tako.forward(torch.rand(3, 2))
```

You can also iterate over the Tako.

```
out = None
for layer in tako.forward_iter(In(torch.rand(3, 2)):
  out = layer.y

print(out)
```


