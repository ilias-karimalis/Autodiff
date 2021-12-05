from autodiff import ad_float

def test_addition():
    x = ad_float(4)
    y = x + 3
    z = 4 + y
    z.backward()
    assert(x.grad == 1.0)
    print("Addition Test Passed")

def some_operations(x, y):
    z = x + y
    v = x * y
    u = v * z
    mu = v * x
    sigma = u * mu
    return sigma


if __name__ == '__main__':
    test_addition()

