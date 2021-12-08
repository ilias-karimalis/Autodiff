import math


###############################################################################
# Auto Diff Float Class
###############################################################################

class ad_float:

    def __init__(self, value: float, compute_graph=None):
        self.value = value
        self.grad = 0.0
        self.compute_graph = compute_graph

    def __str__(self):
        # For better Debug Printing of ad_float class
       return f"ad_float:\nvalue: {self.value}\ngrad: {self.grad}\ncompute_graph: {self.compute_graph}"
    
    def __repr__(self):
        return str(self.value)

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
            compute_graph=ad_substraction_node(self, other)
        )

    def __truediv__(self, other):
        if not isinstance(other, ad_float):
            other = ad_float(other)
        return ad_float(
            self.value / other.value,
            compute_graph=ad_division_node(self, other)
        )

    def __rtruediv__(self, other):
        if not isinstance(other, ad_float):
            other = ad_float(other)
        return ad_float(
            other.value / self.value,
            compute_graph=ad_division_node(other, self)
        )

    def backward(self):
        # Form Compute Graph
        vertices = []
        edges = {}
        generate_graph(self, vertices, edges)

        # Sort The Graph
        sorted_vertices = topological_sort(vertices, edges)

        # Run backward on the graph
        sorted_vertices[0].grad = 1.0
        for vertex in sorted_vertices:
            if vertex.compute_graph is not None:
                vertex.compute_graph.backward(vertex.grad)

###############################################################################
# Auto Diff Tensor Class
###############################################################################

class ad_matrix:

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.__m = [[ad_float(0) for _ in range(width)] for _ in range(height)]

    def __str__(self):
        ret = ""
        for i, row in enumerate(self.__m):
            ret += str(row) 
            ret += "\n" if i < len(self.__m) - 1 else ""
        return ret

    def __getitem__(self, *args):
        # Assert we have the correct number of elements
        assert(len(args) > 0)
        assert(len(args) < 3)
        first, second = args

        if isinstance(first, int) and isinstance(second, int):
            return self.__m[first][second]

        # For now we just crash:
        print(f"We do not yet support matrix array access of the type {type(first)}, {type(second)}.")
        exit(-1)

    def __setitem__(self, args, value):
        assert(len(args) > 0)
        assert(len(args) < 3)
        first, second = args

        # There's better ways of handling this, but this is sufficient for now
        # TODO do better 
        assert(isinstance(value, ad_float))

        if isinstance(first, int) and (second, int):
            self.__m[first][second] = value
            return
        
        # For now we just crash:
        print(f"We do not yet support matrix array access of the type {type(first)}, {type(second)}.")
        exit(-1)
        

def ad_ones(height, width):
    m = ad_matrix(height, width)
    for i in range(height):
        for j in range(width):
            m[i, j] = ad_float(1)
    return m

def ad_zeros(height, width):
    return ad_matrix(height, width)

###############################################################################
# Differentiable Math Functions
###############################################################################

def ad_exp(x: ad_float):
    return ad_float(
        math.exp(x.value),
        compute_graph=ad_exp_node(x)
    )


def ad_log(x: ad_float):
    return ad_float(
        math.log(x.value),
        compute_graph=ad_log_node(x)
    )


def ad_sin(x: ad_float):
    return ad_float(
        math.sin(x.value),
        compute_graph=ad_sin_node(x)
    )


def ad_cos(x: ad_float):
    return ad_float(
        math.cos(x.value),
        compute_graph=ad_cos_node(x)
    )


###############################################################################
# Computational Graph Nodes
###############################################################################

class ad_addition_node:

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.vertices = [left, right]

    def backward(self, acc):
        self.left.grad += acc
        self.right.grad += acc


class ad_substraction_node:

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.vertices
    
    def backward(self, acc):
        self.left.grad += acc
        self.right.grad += -1 * acc


class ad_multiplication_node:

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.vertices = [left, right]

    def backward(self, acc):
        self.left.grad += self.right.value * acc
        self.right.grad += self.left.value * acc


class ad_division_node:

    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator
        self.vertices = [numerator, denominator]

    def backward(self, acc):
        self.numerator.grad += acc / self.denominator.value
        self.denominator.grad += -1 * acc * self.numerator / (self.denominator * self.denominator)


class ad_exp_node:

    def __init__(self, arg):
        self.arg = arg
        self.vertices = [arg]

    def backward(self, acc):
        self.arg.grad += acc * math.exp(self.arg.value)


class ad_log_node:

    def __init__(self, arg):
        self.arg = arg
        self.vertices = [arg]

    def backward(self, acc):
        self.arg.grad += acc / self.arg.value


class ad_sin_node:

    def __init__(self, arg): 
        self.arg = arg
        self.vertices = [arg]

    def backward(self, acc):
        self.arg.grad += acc * math.cos(self.arg.value)


class ad_cos_node:

    def __init__(self, arg):
        self.arg = arg
        self.vertices = [arg]
    
    def backward(self, acc):
        self.arg.grad += -1 * acc * math.sin(self.arg.value)


###############################################################################
# Utility Functions
###############################################################################

# Generates a Topological sorting of the given DAG
def topological_sort(vertices, edges):
    # Mark all the vertices as unvisited
    visited = [False]*len(vertices)
    result = []

    vertex_id_array = {v: i for i, v in enumerate(vertices)}

    # Define Helper Functions
    def get_vertex_id(vertex, vertices):
        for (i, vertex_prime) in enumerate(vertices):
            if vertices is vertex_prime:
                return i
    
    def helper(i, visited_array, result, vertices, edges):
        # We are currently vistiting vertex id
        visited_array[i] = True

        # We now visit all adjacent Vertices that have yet to be visited
        adjacent = edges[vertices[i]]
        for v in adjacent:
            new_i = vertex_id_array[v] #get_vertex_id(v, vertices)
            if not visited_array[new_i]:
                helper(new_i, visited_array, result, vertices, edges)
        
        # Place current node at the front of the result
        result.insert(0, vertices[i])

    # Perform Sort using helper function
    for i in range(len(vertices)):
        if not visited[i]:
            helper(i, visited, result, vertices, edges)
    
    return result


# Generates the computational graph that resulted in the creation of the vertex
def generate_graph(vertex: ad_float, vertices, edges):
    if vertex not in vertices:
        vertices.append(vertex)
        if vertex.compute_graph is not None:
            edges.update({vertex: vertex.compute_graph.vertices})
        else:
            edges.update({vertex: []})
        for v in edges[vertex]:
            generate_graph(v, vertices, edges)
