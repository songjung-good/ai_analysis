from ..contracts import NodeResult
from ..state import GraphState
from ..tools import tools_for


TOOLS = tools_for("startup_discovery")


def run(state: GraphState) -> NodeResult:
    """Collect candidates and select the first startup."""
    raise NotImplementedError
