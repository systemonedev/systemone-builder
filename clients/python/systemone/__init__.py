"""systemone: client for System One decision models such as Kenning.

Ask typed questions about program state; get typed answers with calibrated
probabilities, from a server (:class:`Client`, :class:`AsyncClient`) or in
process from an exported bundle (:class:`Kenning`, needs ``systemone[local]``).
"""

from systemone.client import AsyncClient, Client, SystemOneError
from systemone.types import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Response,
    Score,
    ScoreAnswer,
    Usage,
)

__version__ = "0.1.0"

__all__ = ["AsyncClient", "Choice", "ChoiceAnswer", "Client", "Kenning", "Noul", "NoulAnswer", "Response", "Score",
           "ScoreAnswer", "SystemOneError", "Usage", "__version__"]


def __getattr__(name: str):  # lazy: torch is only needed for local inference
    if name == "Kenning":
        from systemone.local import Kenning

        return Kenning
    raise AttributeError(name)
