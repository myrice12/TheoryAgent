"""Custom exception hierarchy for the TheoryAgent workflow."""


class TheoryAgentError(Exception):
    """Base exception for TheoryAgent runtime failures."""


class StageError(TheoryAgentError):
    """Error in a specific pipeline stage."""
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


class LLMError(TheoryAgentError):
    """Error from LLM interaction (JSON parse failure, empty response, etc.)."""


class ValidationError(TheoryAgentError):
    """Schema or data validation error."""


class ToolError(TheoryAgentError):
    """Error from tool execution in ReAct loop."""


class CheckpointError(TheoryAgentError):
    """Error loading or saving checkpoint data."""
