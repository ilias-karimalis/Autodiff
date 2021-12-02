A working autodiff library

# autodiff

`autodiff` is a toy implementation of [reverse mode automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation).
It was created after I realized that I didn't fully understand how automatic differentiation works.
The entire functionaliry for this library can be found in `autodiff.py`.

## Getting Started
`autodiff` can be obtained by cloning this repo.
The library may then be used by copying the `autodiff.py` file.

## Features/Goals
- [ ] Add implementations for all differentiable python.math functions 
- [ ] Document the `autodiff` library
- [ ] Add extensive tests/examples

## Example Code
```python
def some_operations(x, y):
    z = x + y
    v = x * y
    u = ad_sin(v * z)
    k = v * x
    return ad_log(u * k)

x = ad_float(3)
y = ad_float(4)
ret = some_operations(x, y)
ret.backward()
# x, y will now have updated gradients
```
