from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    
    实现中心差分公式 (f(x+epsilon/2)-f(x-epsilon/2))/epsilon
    """
    tmp_val = list(vals[:])
    tmp_val[arg] += epsilon/2
    forward_value = f(*tmp_val)
    tmp_val[arg] -= epsilon
    backward_value = f(*tmp_val)

    return (forward_value-backward_value) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    res_list = []
    view_var = {}

    def recursive(v: Variable):
        if v.is_constant(): # 遇到常量直接返回
            return
        view_var[v.unique_id] = 1
        res_list.append(v)
        for vp in v.parents: # 递归后续节点
            # 只递归访问没有被遍历过的节点
            if view_var.get(vp.unique_id, 0):
                continue
            recursive(vp)

    recursive(variable)
    return res_list


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    queue = []
    queue.append((variable, deriv))
    while len(queue) != 0:
        (var, d) = queue.pop(0)
        if var.is_leaf():
            var.accumulate_derivative(d)
        else:
            follow_vars = var.chain_rule(d)
            queue.extend(follow_vars)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
