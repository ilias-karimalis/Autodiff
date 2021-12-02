from utils import topological_sort

class ad_addition_node:

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.vertices = [left, right]

    def backward(self, acc):
        self.left.grad += acc
        self.right.grad += acc


class ad_multiplication_node:

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.vertices = [left, right]

    def backward(self, acc):
        self.left.grad += self.right.value * acc
        self.right.grad += self.left.value * acc


class ad_float:

    def __init__(self, value: float, compute_graph=None):
        self.value = value
        self.grad = 0
        self.compute_graph = compute_graph

    def __str__(self):
        return f"ad_float:\nvalue: {self.value}\ngrad: {self.grad}\ncompute_graph: {self.compute_graph}"

    #def __eq__(self, other):
    #return self.value == other.value and self.grad == other.grad and self.compute_graph == other.compute_graph

    def __add__(self, other):
        if not isinstance(other, ad_float):
            other = ad_float(other)
        return ad_float(
            self.value + other.value,
            compute_graph=ad_addition_node(self, other)
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, ad_float):
            other = ad_float(other)
        return ad_float(
            self.value * other.value,
            compute_graph=ad_multiplication_node(self, other)
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if not isinstance(other, ad_float):
            other = ad_float(other)
        return ad_float(
            self.value - other.value,
            # compute_graph=ad_substraction_node(self, other)
        )

    def __backward(self, acc = 1.0):
        print(f"value: {self.value} $$ grad: {self.grad} $$ cg: {self.compute_graph}")
        if self.compute_graph is not None:
            self.compute_graph.backward(acc)

    def backward(self):
        # Form Compute Graph
        vertices = []
        edges = {}
        generate_graph(self, vertices, edges)
        print("Printing Graph:")
        for v in vertices:
            print(v)
        for v in edges.keys():
            print(v)
            print("Is connected to:")
            for n in edges[v]:
                print(n)

        # Sort The Graph
        sorted_vertices = topological_sort(vertices, edges)

        # Run backward on the graph
        sorted_vertices[0].grad = 1.0
        print("Printing Sorted Vertices")
        for vertex in sorted_vertices:
            print(vertex)
            if vertex.compute_graph is not None:
                vertex.compute_graph.backward(vertex.grad)



def generate_graph(vertex, vertices, edges):
    if vertex not in vertices:
        vertices.append(vertex)
        if vertex.compute_graph is not None:
            edges.update({vertex: vertex.compute_graph.vertices})
        else:
            edges.update({vertex: []})
        for v in edges[vertex]:
            generate_graph(v, vertices, edges)





if __name__ == '__main__':
    x = ad_float(3.0)
    y = ad_float(4.0)
    # print(x)
    # print(y)
    z = x + y
    v = x * y
    u = v * z
    mu = v * x
    sigma = u * mu
    # print(z)
    sigma.backward()
    print(f"z.value = {z.value}")
    print(f"v.value = {v.value}")
    print(f"u.value = {u.value}")
    print(f"mu.value = {mu.value}")
    print(f"sigma.value = {sigma.value}")

    print(f"x.grad = {x.grad}")
    print(f"y.grad = {y.grad}")
