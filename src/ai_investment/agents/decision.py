from ..contracts import NodeResult
from ..state import GraphState


def run(state: GraphState) -> NodeResult:
    """Score the startup and append one evaluation."""
    raise NotImplementedError
