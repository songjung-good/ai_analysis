from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str | None = None
    page: int | None = Field(default=None, ge=1)
    excerpt: str = Field(min_length=1)
    published_at: str | None = None
