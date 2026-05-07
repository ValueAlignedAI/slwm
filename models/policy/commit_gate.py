"""Policy/commitment gate shape contract."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from models.module import ShapeModule
from models.types import ensure_latent


def _proposal_items(proposals: Mapping[str, Any] | Iterable[Any]) -> list[dict[str, Any]]:
    if isinstance(proposals, Mapping):
        raw_items = proposals.values()
    else:
        raw_items = proposals
    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, Mapping) and "proposal" in item:
            normalized.append(dict(item["proposal"]))
        elif isinstance(item, Mapping):
            normalized.append(dict(item))
        else:
            raise TypeError(f"Unsupported proposal item {type(item)!r}")
    return normalized


class PolicyCommitGate(ShapeModule):
    """Deterministic I0 policy stub.

    Forward input shape:
        ``z_world: FloatTensor[B,T,D]`` plus proposal dictionaries.

    Forward output contract:
        ``{"commitments": list, "gates": dict, "noop_probability": float,
        "metadata": dict}``.

    I0 does not learn or optimize policy behavior. By default it commits to
    no-op to avoid confusing proposal heads with external behavior. A test or
    caller may pass ``goal={"commit_head": "text_decoder"}`` to exercise the
    single-head commitment path.
    """

    def forward(
        self,
        z_world: Any,
        proposals: Mapping[str, Any] | Iterable[Any],
        uncertainty: Any | None = None,
        goal: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        b, t, d = ensure_latent(z_world, "z_world")
        proposal_list = _proposal_items(proposals)
        requested_head = None if goal is None else goal.get("commit_head")

        gates = {str(proposal.get("head", f"proposal_{idx}")): 0.0 for idx, proposal in enumerate(proposal_list)}
        commitments: list[dict[str, Any]] = []

        if requested_head is not None:
            for proposal in proposal_list:
                if proposal.get("head") == requested_head:
                    gates[str(requested_head)] = 1.0
                    committed = dict(proposal)
                    committed.update({"committed": True, "status": "committed"})
                    commitments.append(committed)
                    break

        noop_probability = 0.0 if commitments else 1.0
        if not commitments:
            gates["noop"] = 1.0
            commitments.append(
                {
                    "head": "noop",
                    "channel": "none",
                    "committed": True,
                    "status": "committed",
                    "intention": "observe_wait",
                    "reason": "i0_policy_stub_default_noop",
                }
            )

        return {
            "commitments": commitments,
            "gates": gates,
            "noop_probability": noop_probability,
            "metadata": {
                "policy": self.__class__.__name__,
                "implementation": "i0_shape_contract_stub",
                "proposal_count": len(proposal_list),
                "z_world_shape": [b, t, d],
                "uncertainty_provided": uncertainty is not None,
            },
        }
