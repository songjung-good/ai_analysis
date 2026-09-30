"""Structured output contract for the deployment and business Agent."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


Stage = Literal["demo", "pilot", "paid_operation", "unknown"]
Text = Annotated[str, Field(min_length=1)]


class BusinessModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CustomerCase(BusinessModel):
    customer_name: Text
    use_case: Text
    operation_stage: Stage
    is_paid: bool | None = Field(strict=True)
    source_ids: list[Text] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_paid_operation(self):
        if self.operation_stage == "paid_operation" and self.is_paid is not True:
            raise ValueError("paid_operation requires confirmed payment")
        return self


class DeploymentCost(BusinessModel):
    cost_type: Literal["purchase", "subscription", "installation", "integration", "training", "maintenance", "other"]
    description: Text
    amount: Annotated[FiniteFloat, Field(ge=0)] | None
    currency: Text | None
    period: Text | None
    scope: Text
    source_ids: list[Text] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_currency(self):
        if self.amount is not None and self.currency is None:
            raise ValueError("a known cost amount requires currency")
        return self


class DeploymentEffect(BusinessModel):
    metric: Text
    value: FiniteFloat | None
    unit: Text | None
    baseline: Text | None
    measurement_conditions: Text | None
    description: Text
    source_ids: list[Text] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unit(self):
        if self.value is not None and self.unit is None:
            raise ValueError("a known effect value requires unit")
        return self


class RegulatoryRisk(BusinessModel):
    risk: Text
    jurisdiction: Text | None
    certification_status: Text | None
    follow_up: Text
    source_ids: list[Text] = Field(min_length=1)


class EvidenceAssessment(BusinessModel):
    claim: Text
    level: Literal["cross_verified", "single_source", "claim_only", "unknown", "conflicting"]
    reason: Text
    source_ids: list[Text]

    @model_validator(mode="after")
    def validate_sources(self):
        if self.level != "unknown" and not self.source_ids:
            raise ValueError("an assessed claim requires sources unless unknown")
        return self


class BusinessAnalysis(BusinessModel):
    commercialization_stage: Stage
    stage_reason: Text
    stage_source_ids: list[Text]
    customer_cases: list[CustomerCase]
    deployment_costs: list[DeploymentCost]
    deployment_effects: list[DeploymentEffect]
    regulatory_risks: list[RegulatoryRisk]
    evidence_assessment: list[EvidenceAssessment]
    information_gaps: list[Text]
    summary: Text

    @model_validator(mode="after")
    def validate_stage(self):
        if self.commercialization_stage != "unknown" and not self.stage_source_ids:
            raise ValueError("a known commercialization stage requires sources")
        if self.commercialization_stage == "paid_operation" and not any(
            case.operation_stage == "paid_operation" and case.is_paid is True
            for case in self.customer_cases
        ):
            raise ValueError("paid_operation requires a confirmed paid customer case")
        return self
