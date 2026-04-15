"""Pydantic data models for TheoryAgent."""

from theoryagent.schemas.ideation import (
    GapAnalysis,
    Hypothesis,
    IdeationOutput,
    PaperReference,
)
from theoryagent.schemas.experiment import (
    AblationGroup,
    AblationResult,
    Baseline,
    Dataset,
    ExperimentBlueprint,
    ExperimentResults,
    Metric,
    MetricResult,
    MethodResult,
    TrainingLogEntry,
)
from theoryagent.schemas.writing import (
    WritingOutput,
)
from theoryagent.schemas.figure import (
    FigureOutput,
    FigureRecord,
)
from theoryagent.schemas.paper import (
    FigurePlaceholder,
    PaperSkeleton,
    Section,
)
from theoryagent.schemas.manifest import (
    ArtifactRecord,
    PipelineStage,
    StageRecord,
    WorkspaceManifest,
)
from theoryagent.schemas.evidence import (
    EvidenceBundle,
    ExtractedMetric,
)
from theoryagent.schemas.review import (
    ConsistencyIssue,
    ReviewOutput,
    SectionReview,
)
from theoryagent.schemas.iteration import (
    ExperimentHypothesis,
    FeedbackAnalysis,
    IterationState,
    PreflightReport,
    PreflightResult,
    RoundResult,
    TrainingDynamics,
)

__all__ = [
    "GapAnalysis",
    "Hypothesis",
    "IdeationOutput",
    "PaperReference",
    "AblationGroup",
    "Baseline",
    "Dataset",
    "ExperimentBlueprint",
    "Metric",
    "FigurePlaceholder",
    "PaperSkeleton",
    "Section",
    "ArtifactRecord",
    "PipelineStage",
    "StageRecord",
    "WorkspaceManifest",
    "EvidenceBundle",
    "ExtractedMetric",
    "ConsistencyIssue",
    "ReviewOutput",
    "SectionReview",
    "ExperimentHypothesis",
    "FeedbackAnalysis",
    "IterationState",
    "PreflightReport",
    "PreflightResult",
    "RoundResult",
    "TrainingDynamics",
    "AblationResult",
    "ExperimentResults",
    "MetricResult",
    "MethodResult",
    "TrainingLogEntry",
    "WritingOutput",
    "FigureOutput",
    "FigureRecord",
]
