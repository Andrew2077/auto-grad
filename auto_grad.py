from uuid import uuid4
from graphviz import Digraph
import numpy as np
import functions as fn


class CompNode:
    def __init__(self, val: float, children: list = [], op: str = "assign"):
        self.val = val
        self.children = children
        self.op = op
        # * on instantiation, grad is set to 0
        self.grad = 0
        # *backward function
        self.backward_fn = lambda: None
        self.identiy = uuid4()

    def __to_node(self, obj):
        if not isinstance(obj, CompNode):
            if isinstance(obj, list):
                return CompNode(obj[0])
            else:
                return CompNode(obj)
        else:
            return obj

    ############################## ADDITION ##################################
    def __add__(self, other):
        other = self.__to_node(other)
        out = CompNode(val=self.val + other.val, children=[self, other], op="add")

        def backwardprop():
            self.grad += out.grad * 1
            other.grad += out.grad * 1

        out.backward_fn = backwardprop
        return out

    def __radd__(self, other):
        return self.__add__(other)

    ############################## SUBTRACTION ################################
    def __sub__(self, other):
        other = self.__to_node(other)
        out = CompNode(val=self.val - other.val, children=[self, other], op="sub")

        def backwardprop():
            # print(f"self.grad: {self.grad} | out.grad: {out.grad}")
            self.grad += out.grad * 1
            other.grad += out.grad * -1

        out.backward_fn = backwardprop
        return out

    def __rsub__(self, other):
        other = self.__to_node(other)
        return other - self

    ############################## MULTIPLICATION ################################
    def __mul__(self, other):
        other = self.__to_node(other)
        out = CompNode(val=self.val * other.val, children=[self, other], op="mul")

        def backwardprop():
            self.grad += out.grad * other.val
            other.grad += out.grad * self.val

        out.backward_fn = backwardprop
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    ############################## POWER ################################
    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be a number")
        out = CompNode(val=self.val**exponent, children=[self], op="pow")

        def backward_prop():
            self.grad += out.grad * (exponent * self.val ** (exponent - 1))

        out.backward_fn = backward_prop
        return out

    def __rpow__(self, other):
        other = self.__to_node(other)
        return other**self

    ############################## DIVISION ################################
    def __truediv__(self, other):
        other = self.__to_node(other)
        out = CompNode(val=self.val / other.val, children=[self, other], op="div")

        def _backward_prop():
            self.grad += out.grad * (1 / other.val)
            other.grad += out.grad * (-self.val / (other.val**2))

        out.backward_fn = _backward_prop
        return out

    def __rtruediv__(self, other):
        other = self.__to_node(other)
        return other / self

    ############################## LOG ################################
    def log(self):
        out = CompNode(val=np.log(self.val), children=[self], op="ln")

        def _backward_prop():
            self.grad += out.grad * (1 / self.val)

        out.backward_fn = _backward_prop
        return out

    ############################## Equality ################################
    def __eq__(self, other):
        other = self.__to_node(other)
        return self.val == other.val

    def __req__(self, other):
        other = self.__to_node(other)
        return self.__eq__(other)

    def __repr__(self):
        return f"Op: {self.op}, Val: {self.val:.5f}, children: {len(self.children)}, grad: {self.grad:.5f}"

    # * end of operations
    ############################## Hash ################################
    def __hash__(self):
        return int(self.identiy)

    ############################## Topological Sort ################################
    def toposort(self, collect_edges=False):
        res = []
        visited = set()
        if collect_edges:
            edges = []

        def visit(node):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    if collect_edges:
                        edges.append((child, node))
                    visit(child)
                res.append(node)

        visit(self)
        if collect_edges:
            return res, edges
        return res

    ############################## Draw Graph ################################
    def draw_graph(self):
        nodes, edges = self.toposort(collect_edges=True)
        dot = Digraph(format="svg", graph_attr={"rankdir": "TB"})
        for idx, n in enumerate(nodes[::-1]):
            dot.node(
                name=str(hash(n)),
                label=f"Op: {idx} : {n.op}\nVal : {n.val:.2f}\nGrad : {n.grad:.2f}",
            )
        for n1, n2 in edges:
            dot.edge(str(hash(n1)), str(hash(n2)))
        return dot

    ############################## Backward ################################
    def backward(self):
        nodes = self.toposort()
        self.grad = 1
        for node in reversed(nodes):
            node.backward_fn()
