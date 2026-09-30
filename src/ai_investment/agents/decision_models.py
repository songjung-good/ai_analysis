"""Investment score evidence contract; decisions are computed in Python."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


Text = Annotated[str, Field(min_length=1)]


class DecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CriterionScore(DecisionModel):
    score: Annotated[FiniteFloat, Field(ge=1, le=5, strict=True)] | None
    reason: Text
    source_ids: list[Text]

    @model_validator(mode="after")
    def require_evidence(self):
        if self.score is not None and not self.source_ids:
            raise ValueError("a numeric score requires evidence source IDs")
        return self


class BlockingRisk(DecisionModel):
    category: Literal["legal", "safety"]
    reason: Text
    source_ids: list[Text] = Field(min_length=1)


class InvestmentAssessment(DecisionModel):
    team: CriterionScore
    market: CriterionScore
    technology: CriterionScore
    competition: CriterionScore
    traction: CriterionScore
    investment_terms: CriterionScore
    blocking_risks: list[BlockingRisk]
    critical_information_gaps: list[Text]
