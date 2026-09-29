"""The concrete streaming sources used by Fraise, in training order."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    repo_id: str
    config: str | None = None
    revision: str | None = None
    size_gb: float | None = None


FINEWEB = Source(
    "fineweb-edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", size_gb=10.05
)
KHAN = Source("cosmopedia-khanacademy", "HuggingFaceTB/cosmopedia", "khanacademy")
STORIES = Source("cosmopedia-stories", "HuggingFaceTB/cosmopedia", "stories")
STREAM_SOURCES = {source.name: source for source in (FINEWEB, KHAN, STORIES)}
SOURCE_NAMES = ("babylm", *STREAM_SOURCES)
VALIDATION_SIZE_GB = 0.05
