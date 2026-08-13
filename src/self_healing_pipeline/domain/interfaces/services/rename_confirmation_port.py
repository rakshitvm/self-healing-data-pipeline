"""LLM rename-confirmation abstraction — the only LLM call in schema repair.

Used exclusively by the nested rename-resolution subgraph's `llm_confirm`
node: given a deterministic `RenameHint` (already computed by name
similarity — the LLM never computes the raw diff itself), ask the LLM
whether it genuinely believes this is a rename versus an unrelated
add+drop, and how confident it is. Unlike `CsvRepairProposalPort`, this
contract deliberately surfaces token usage, so real cost/usage can be
recorded on the migration history and MLflow run metrics (see Tier 2
requirement 9) instead of being discarded.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from self_healing_pipeline.domain.value_objects.column_diff import RenameHint


class RenameConfirmation(BaseModel):
    """The LLM's judgment on a single `RenameHint`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    confirmed: bool
    llm_confidence: float = Field(ge=0.0, le=1.0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


@runtime_checkable
class RenameConfirmationPort(Protocol):
    """Confirms or rejects a single deterministic rename candidate."""

    def confirm(self, *, table: str, hint: RenameHint) -> RenameConfirmation:
        """Ask the LLM to confirm/reject `hint` for `table`."""
        ...
