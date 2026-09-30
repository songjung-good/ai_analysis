from ..contracts import NodeResult
from ..state import GraphState
from ..tools import tools_for


TOOLS = tools_for("technical_analysis")


def run(state: GraphState) -> NodeResult:
    """Validate core technology, evidence, and limitations with CRAG."""
    raise NotImplementedError
