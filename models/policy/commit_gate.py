"""Sprint I3 policy/commitment gate stubs.

The gate consumes output-head proposals and marks every proposal as one of:

``committed``
    A policy-selected external behavior.

``suppressed``
    A valid proposal that was not selected for external output.

``diagnostic-only``
    An internal/probe output that may be inspected but must not become external
    behavior through the default policy.

The implementation is deterministic and dependency-light.  It is a routing API
stub, not a trained behavior-quality claim.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from data.contract import MODALITY_IDS
from models.module import ShapeModule
from models.types import ensure_latent


def _proposal_items(proposals: Mapping[str, Any] | Iterable[Any]) -> list[dict[str, Any]]:
    """Normalize proposal outputs or raw proposal dictionaries.

    Accepted item forms:
        ``{"proposal": {...}, ...}`` as returned by output heads, or a raw
        proposal mapping.  Non-mapping values are rejected so callers do not
        accidentally route arbitrary objects as external behavior.
    """

    if isinstance(proposals, Mapping):
        raw_items = proposals.values()
    else:
        raw_items = proposals
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        if isinstance(item, Mapping) and "proposal" in item:
            proposal = dict(item["proposal"])
        elif isinstance(item, Mapping):
            proposal = dict(item)
        else:
            raise TypeError(f"Unsupported proposal item {type(item)!r}")
        proposal.setdefault("head", f"proposal_{idx}")
        proposal.setdefault("channel", "internal")
        proposal.setdefault("score", 0.0)
        proposal.setdefault("committed", False)
        proposal.setdefault("status", "proposal")
        proposal.setdefault("diagnostic_only", False)
        proposal.setdefault("internal_only", bool(proposal.get("diagnostic_only", False)))
        proposal.setdefault("metadata", {})
        normalized.append(proposal)
    return normalized


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def _goal_requested_heads(goal: Mapping[str, Any]) -> tuple[bool, list[str]]:
    if "commit_heads" in goal:
        return True, _as_list(goal.get("commit_heads"))
    if "commit_head" in goal:
        return True, _as_list(goal.get("commit_head"))
    return False, []


def _goal_requested_channels(goal: Mapping[str, Any]) -> tuple[bool, list[str]]:
    if "commit_channels" in goal:
        return True, _as_list(goal.get("commit_channels"))
    if "commit_channel" in goal:
        return True, _as_list(goal.get("commit_channel"))
    return False, []


def _is_noop(proposal: Mapping[str, Any]) -> bool:
    return proposal.get("head") == "noop" or proposal.get("channel") in {"none", "noop"} or proposal.get("modality") == "noop"


def _annotate(proposal: Mapping[str, Any], *, status: str, committed: bool, reason: str) -> dict[str, Any]:
    annotated = dict(proposal)
    annotated["status"] = status
    annotated["committed"] = bool(committed)
    annotated["reason"] = reason
    annotated["decision"] = status
    return annotated


def _synthetic_noop(reason: str) -> dict[str, Any]:
    return {
        "proposal_id": "noop:none",
        "head": "noop",
        "channel": "none",
        "modality": "noop",
        "modality_id": MODALITY_IDS["noop"],
        "score": 1.0,
        "source_tag": "unknown",
        "diagnostic_only": False,
        "internal_only": False,
        "committed": True,
        "status": "committed",
        "decision": "committed",
        "intention": "observe_wait",
        "reason": reason,
        "metadata": {"synthesized_by_policy": True},
    }


class FixedRulePolicyCommitGate(ShapeModule):
    """Deterministic fixed-rule policy baseline for Sprint I3.

    Forward input shape:
        ``z_world: FloatTensor[B,T,D]`` plus proposal dictionaries.

    Forward output contract:
        ``{"commitments": list, "suppressed": list, "diagnostic_only": list,
        "decisions": list, "gates": dict, "noop_probability": float,
        "metadata": dict}``.

    Default behavior is safety preserving: no external text/audio/visual output
    is committed unless the caller requests it via ``goal`` or enables score
    routing.  If no external head is selected, the gate commits a valid no-op.
    """

    implementation = "i3_fixed_rule_policy_stub"

    def __init__(
        self,
        *,
        score_threshold: float = 0.5,
        commit_by_score: bool = False,
        allow_diagnostic_commit: bool = False,
        allow_noop_with_external: bool = False,
    ) -> None:
        self.score_threshold = float(score_threshold)
        self.commit_by_score = bool(commit_by_score)
        self.allow_diagnostic_commit = bool(allow_diagnostic_commit)
        self.allow_noop_with_external = bool(allow_noop_with_external)

    def _candidate_selected(
        self,
        proposal: Mapping[str, Any],
        *,
        goal: Mapping[str, Any],
        heads_requested: bool,
        requested_heads: Sequence[str],
        channels_requested: bool,
        requested_channels: Sequence[str],
        allow_by_score: bool,
        score_threshold: float,
    ) -> tuple[bool, str]:
        head = str(proposal.get("head", ""))
        channel = str(proposal.get("channel", ""))
        if heads_requested:
            return head in requested_heads, "requested_head" if head in requested_heads else "head_not_requested"
        if channels_requested:
            return channel in requested_channels, "requested_channel" if channel in requested_channels else "channel_not_requested"
        if bool(goal.get("force_noop", False)):
            return _is_noop(proposal), "force_noop"
        if allow_by_score:
            score = float(proposal.get("score", 0.0))
            return score >= score_threshold, "score_threshold_met" if score >= score_threshold else "score_below_threshold"
        return False, "default_no_external_commit"

    def forward(
        self,
        z_world: Any,
        proposals: Mapping[str, Any] | Iterable[Any],
        uncertainty: Any | None = None,
        goal: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        b, t, d = ensure_latent(z_world, "z_world")
        proposal_list = _proposal_items(proposals)
        goal_map = dict(goal or {})
        mode = str(goal_map.get("mode", "commit"))
        heads_requested, requested_heads = _goal_requested_heads(goal_map)
        channels_requested, requested_channels = _goal_requested_channels(goal_map)
        score_threshold = float(goal_map.get("score_threshold", self.score_threshold))
        allow_by_score = bool(goal_map.get("allow_by_score", self.commit_by_score))
        allow_diagnostic_commit = bool(goal_map.get("allow_diagnostic_commit", self.allow_diagnostic_commit))
        allow_noop_with_external = bool(goal_map.get("allow_noop_with_external", self.allow_noop_with_external))
        max_commitments = goal_map.get("max_commitments")
        max_commitments_int = None if max_commitments is None else max(0, int(max_commitments))
        commit_noop = bool(goal_map.get("commit_noop", True))

        commitments: list[dict[str, Any]] = []
        suppressed: list[dict[str, Any]] = []
        diagnostic_only: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        gates: dict[str, float] = {str(proposal.get("head")): 0.0 for proposal in proposal_list}
        noop_proposals: list[dict[str, Any]] = []
        deferred_noop_proposals: list[dict[str, Any]] = []

        for proposal in proposal_list:
            head = str(proposal.get("head"))
            channel = str(proposal.get("channel", ""))
            is_noop = _is_noop(proposal)
            if is_noop:
                noop_proposals.append(proposal)

            is_diagnostic = bool(proposal.get("diagnostic_only", False) or proposal.get("internal_only", False) or mode == "explore")
            if is_diagnostic and not allow_diagnostic_commit:
                annotated = _annotate(proposal, status="diagnostic-only", committed=False, reason="diagnostic_output_not_external")
                diagnostic_only.append(annotated)
                decisions.append(annotated)
                gates[head] = 0.0
                continue

            explicit_noop_request = (
                (heads_requested and head in requested_heads)
                or (channels_requested and channel in requested_channels)
                or bool(goal_map.get("force_noop", False))
            )
            if is_noop and not explicit_noop_request:
                deferred_noop_proposals.append(proposal)
                gates[head] = 0.0
                continue

            selected, reason = self._candidate_selected(
                proposal,
                goal=goal_map,
                heads_requested=heads_requested,
                requested_heads=requested_heads,
                channels_requested=channels_requested,
                requested_channels=requested_channels,
                allow_by_score=allow_by_score,
                score_threshold=score_threshold,
            )
            if is_noop and allow_by_score and not heads_requested and not channels_requested and not bool(goal_map.get("force_noop", False)):
                selected = False
                reason = "noop_deferred_until_no_external_commit"
            external_commits_so_far = [item for item in commitments if not _is_noop(item)]
            if is_noop and external_commits_so_far and not allow_noop_with_external:
                selected = False
                reason = "noop_suppressed_because_external_committed"
            if selected and max_commitments_int is not None and len(commitments) >= max_commitments_int:
                selected = False
                reason = "max_commitments_reached"

            if selected:
                annotated = _annotate(proposal, status="committed", committed=True, reason=reason)
                commitments.append(annotated)
                decisions.append(annotated)
                gates[head] = 1.0
            else:
                annotated = _annotate(proposal, status="suppressed", committed=False, reason=reason)
                suppressed.append(annotated)
                decisions.append(annotated)
                gates[head] = 0.0

        external_commitments = [item for item in commitments if not _is_noop(item)]
        noop_commitments = [item for item in commitments if _is_noop(item)]
        if external_commitments and noop_commitments and not allow_noop_with_external:
            kept_commitments: list[dict[str, Any]] = []
            replacement_by_id: dict[int, dict[str, Any]] = {}
            for commitment in commitments:
                if _is_noop(commitment):
                    suppressed_noop = _annotate(
                        commitment,
                        status="suppressed",
                        committed=False,
                        reason="noop_suppressed_because_external_committed",
                    )
                    suppressed.append(suppressed_noop)
                    replacement_by_id[id(commitment)] = suppressed_noop
                    gates[str(commitment.get("head", "noop"))] = 0.0
                else:
                    kept_commitments.append(commitment)
            decisions = [replacement_by_id.get(id(item), item) for item in decisions]
            commitments = kept_commitments
            noop_commitments = []

        if deferred_noop_proposals:
            if external_commitments or not commit_noop or mode == "explore":
                reason = "noop_suppressed_because_external_committed" if external_commitments else "noop_not_committed"
                for proposal in deferred_noop_proposals:
                    annotated = _annotate(proposal, status="suppressed", committed=False, reason=reason)
                    suppressed.append(annotated)
                    decisions.append(annotated)

        if not external_commitments and not noop_commitments and commit_noop and mode != "explore":
            if deferred_noop_proposals:
                noop_commitment = _annotate(
                    deferred_noop_proposals[0], status="committed", committed=True, reason="no_external_commit_default_noop"
                )
            elif noop_proposals:
                noop_commitment = _annotate(noop_proposals[0], status="committed", committed=True, reason="no_external_commit_default_noop")
            else:
                noop_commitment = _synthetic_noop("no_external_commit_default_noop")
            commitments.append(noop_commitment)
            decisions.append(noop_commitment)
            gates["noop"] = 1.0
            noop_commitments = [noop_commitment]

        noop_probability = 1.0 if noop_commitments or not external_commitments else 0.0
        return {
            "commitments": commitments,
            "suppressed": suppressed,
            "diagnostic_only": diagnostic_only,
            "decisions": decisions,
            "gates": gates,
            "noop_probability": noop_probability,
            "metadata": {
                "policy": self.__class__.__name__,
                "implementation": self.implementation,
                "proposal_count": len(proposal_list),
                "decision_count": len(decisions),
                "committed_count": len(commitments),
                "external_commitment_count": len(external_commitments),
                "suppressed_count": len(suppressed),
                "diagnostic_only_count": len(diagnostic_only),
                "z_world_shape": [b, t, d],
                "uncertainty_provided": uncertainty is not None,
                "goal": goal_map,
                "mode": mode,
            },
        }


class PolicyCommitGate(FixedRulePolicyCommitGate):
    """Default policy gate export, currently the fixed-rule I3 baseline."""


class LearnedPolicyCommitGateStub(FixedRulePolicyCommitGate):
    """Interface-compatible learned-policy placeholder.

    This class intentionally has no trainable parameters and does not optimize
    behavior.  It records deterministic pseudo-logits derived from proposal
    scores, then delegates to the fixed-rule routing semantics so callers can
    wire a future learned policy without changing the API.
    """

    implementation = "i3_learned_policy_stub_no_training"

    def forward(
        self,
        z_world: Any,
        proposals: Mapping[str, Any] | Iterable[Any],
        uncertainty: Any | None = None,
        goal: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        proposal_list = _proposal_items(proposals)
        policy_logits = {str(proposal.get("head")): float(proposal.get("score", 0.0)) for proposal in proposal_list}
        result = super().forward(z_world, proposal_list, uncertainty=uncertainty, goal=goal)
        result["metadata"]["policy_logits"] = policy_logits
        result["metadata"]["learned_policy_stub"] = True
        result["metadata"]["training_enabled"] = False
        return result


__all__ = ["FixedRulePolicyCommitGate", "LearnedPolicyCommitGateStub", "PolicyCommitGate", "_proposal_items"]
