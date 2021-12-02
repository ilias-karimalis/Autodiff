from autodiff import ad_float

if __name__ == '__main__':
    x = ad_float(3.0)
    y = ad_float(4.0)
    z = x + y
    v = x * y
    u = v * z
    mu = v * x
    sigma = u * mu
    sigma.backward()
    print(f"z.value = {z.value}")
    print(f"v.value = {v.value}")
    print(f"u.value = {u.value}")
    print(f"mu.value = {mu.value}")
    print(f"sigma.value = {sigma.value}")

    print(f"x.grad = {x.grad}")
    print(f"y.grad = {y.grad}")

