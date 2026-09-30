from ..contracts import NodeResult
from ..state import GraphState
from ..tools import tools_for


TOOLS = tools_for("customer_profile")


def run(state: GraphState) -> NodeResult:
    """Classify subdomain, paying customer, and customer problem."""
    raise NotImplementedError
