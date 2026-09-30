from ..contracts import NodeResult
from ..state import GraphState
from ..tools import tools_for


TOOLS = tools_for("market_analysis")


def run(state: GraphState) -> NodeResult:
    """Assess market opportunity and competitors with CRAG."""
    raise NotImplementedError
