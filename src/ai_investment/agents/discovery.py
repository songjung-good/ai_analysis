from ..contracts import NodeResult
from ..state import GraphState


def run(state: GraphState) -> NodeResult:
    """Collect candidates and select the first startup."""
    raise NotImplementedError
