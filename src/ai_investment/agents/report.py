from ..contracts import NodeResult
from ..state import GraphState
from ..tools import tools_for


TOOLS = tools_for("report_generation")


def run(state: GraphState) -> NodeResult:
    """Create the final report from evaluations and cited references."""
    raise NotImplementedError
