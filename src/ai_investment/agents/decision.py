from ..contracts import NodeResult
from ..scoring import calculate_score
from ..state import GraphState
from ..tools import tools_for


TOOLS = tools_for("investment_decision")

__all__ = ["TOOLS", "calculate_score", "run"]


def run(state: GraphState) -> NodeResult:
    """Score the startup and append one evaluation."""
    raise NotImplementedError
