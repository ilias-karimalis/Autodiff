from autodiff import ad_float

def some_operations(x, y):
    z = x + y
    v = x * y
    u = v * z
    mu = v * x
    sigma = u * mu
    return sigma


if __name__ == '__main__':
    x = ad_float(3.0)
    y = ad_float(4.0)
    sigma = some_operations(x, y)
    sigma.backward()
    print(f"sigma.value = {sigma.value}")

    print(f"x.grad = {x.grad}")
    print(f"y.grad = {y.grad}")

