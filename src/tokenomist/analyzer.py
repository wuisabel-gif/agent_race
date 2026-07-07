"""Turn conversations into structured traffic traces and aggregate metrics.

Two outputs are produced from a :class:`~tokenomist.models.Conversation`:

* a per-turn **traffic trace** (:class:`TraceRow`) — the workload-characterization
  view, where each assistant generation re-reads the growing context, so input
  tokens accumulate exactly as they would against a real API; and
* an aggregate **agent report** (:class:`AgentReport`) — the dashboard view, with
  cost, latency, retries, corrections, and convergence efficiency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .models import Conversation, Role, Turn
from .pricing import PriceBook
from .tokens import estimate_tokens


@dataclass
class TraceRow:
    """One row of the structured traffic trace for a single turn."""

    task_id: str
    agent: str
    model: str | None
    turn_index: int
    role: str
    content: str
    content_length: int
    input_tokens: int
    output_tokens: int
    usage_details: dict[str, int]
    provided_usage_details: dict[str, int]
    tool_calls: int
    tool_failures: int
    latency_ms: float
    cost_usd: float | None
    cost_details: dict[str, float]
    provided_cost_details: dict[str, float]
    cumulative_cost_usd: float | None
    is_retry: bool
    is_correction: bool


@dataclass
class AgentReport:
    """Aggregate metrics for one agent run, ready for ranking/display."""

    agent: str
    task_id: str
    model: str | None
    provider: str | None
    turn_count: int
    assistant_turns: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    usage_details: dict[str, int]
    provided_usage_details: dict[str, int]
    tool_calls: int
    tool_success_rate: float
    retry_loops: int
    correction_count: int
    latency_estimate_ms: float
    cost_estimate_usd: float | None
    cost_details: dict[str, float]
    provided_cost_details: dict[str, float]
    success_turn: int | None
    turns_to_success: int | None
    tokens_to_success: int | None
    final_correct: bool
    final_score: float
    convergence_efficiency: float
    trace: list[TraceRow] = field(default_factory=list)

    def summary_dict(self) -> dict:
        """Report fields without the (potentially large) trace rows."""

        data = asdict(self)
        data.pop("trace", None)
        return data


def _turn_output_tokens(turn: Turn) -> int:
    """Output tokens generated in a turn: its text plus serialized tool args."""

    if turn.output_tokens is not None:
        return turn.output_tokens
    total = estimate_tokens(turn.content)
    for call in turn.tool_calls:
        total += estimate_tokens(call.name)
        total += estimate_tokens(str(call.arguments))
    return total


def _turn_context_tokens(turn: Turn) -> int:
    """Token footprint a turn adds to the running context window."""

    total = estimate_tokens(turn.content)
    for call in turn.tool_calls:
        total += estimate_tokens(call.name) + estimate_tokens(str(call.arguments))
        if call.result:
            total += estimate_tokens(str(call.result))
    return total


def build_trace(conv: Conversation, prices: PriceBook | None = None) -> list[TraceRow]:
    """Produce the per-turn traffic trace for ``conv``.

    Each assistant turn is billed for the full prior context as input (mirroring
    how stateless chat APIs re-send history) plus its own generated tokens as
    output. Non-assistant turns carry no independent cost; their tokens are
    folded into the next assistant turn's input.
    """

    prices = prices or PriceBook()
    # A conversation has one model, so its cost is uniformly known or unknown.
    # When unknown we surface ``None`` (n/a) rather than a fabricated number.
    model_known = prices.resolve(conv.model) is not None
    rows: list[TraceRow] = []
    running_context = 0
    cumulative_cost: float | None = 0.0 if model_known else None

    for turn in conv.turns:
        tool_calls = len(turn.tool_calls)
        tool_failures = sum(1 for c in turn.tool_calls if not c.ok)

        if turn.role is Role.ASSISTANT:
            in_toks = turn.input_tokens if turn.input_tokens is not None else running_context
            out_toks = _turn_output_tokens(turn)
            usage_details = {
                **turn.usage_details,
                "input_tokens": turn.usage_details.get("input_tokens", in_toks),
                "output_tokens": turn.usage_details.get("output_tokens", out_toks),
            }
            provided_usage_details = dict(turn.provided_usage_details)
            provided_cost_details = _finalize_cost_details(dict(turn.provided_cost_details))
            if provided_cost_details:
                cost_details = provided_cost_details
            elif turn.cost_details:
                cost_details = _finalize_cost_details(dict(turn.cost_details))
            else:
                cost_details = prices.cost_details_usd(conv.model, usage_details)
            cost = None if cost_details is None else _total_cost_from_details(cost_details)
            latency = (
                turn.latency_ms
                if turn.latency_ms is not None
                else prices.latency_ms(conv.model, out_toks)
            )
        else:
            in_toks, out_toks, latency = 0, 0, 0.0
            cost = 0.0 if model_known else None
            usage_details = {}
            provided_usage_details = {}
            cost_details = {} if model_known else None
            provided_cost_details = {}

        if cost is not None and cumulative_cost is not None:
            cumulative_cost += cost
        rows.append(
            TraceRow(
                task_id=conv.task_id,
                agent=conv.agent,
                model=conv.model,
                turn_index=turn.index,
                role=turn.role.value,
                content=turn.content or "",
                content_length=len(turn.content or ""),
                input_tokens=in_toks,
                output_tokens=out_toks,
                usage_details=usage_details,
                provided_usage_details=provided_usage_details,
                tool_calls=tool_calls,
                tool_failures=tool_failures,
                latency_ms=round(latency, 1),
                cost_usd=None if cost is None else round(cost, 6),
                cost_details=(
                    {}
                    if cost_details is None
                    else {k: round(v, 6) for k, v in cost_details.items()}
                ),
                provided_cost_details={k: round(v, 6) for k, v in provided_cost_details.items()},
                cumulative_cost_usd=(
                    None if cumulative_cost is None else round(cumulative_cost, 6)
                ),
                is_retry=turn.is_retry,
                is_correction=turn.is_correction,
            )
        )
        running_context += _turn_context_tokens(turn)

    return rows


def analyze(conv: Conversation, prices: PriceBook | None = None) -> AgentReport:
    """Compute the aggregate :class:`AgentReport` for ``conv``."""

    prices = prices or PriceBook()
    trace = build_trace(conv, prices)

    input_tokens = sum(r.input_tokens for r in trace)
    output_tokens = sum(r.output_tokens for r in trace)
    usage_details = _sum_int_maps(r.usage_details for r in trace)
    provided_usage_details = _sum_int_maps(r.provided_usage_details for r in trace)
    cost_details = _sum_float_maps(r.cost_details for r in trace)
    provided_cost_details = _sum_float_maps(r.provided_cost_details for r in trace)
    # None if any turn's price is unknown (uniform per conversation).
    total_cost: float | None = (
        None if any(r.cost_usd is None for r in trace) else sum(r.cost_usd for r in trace)  # type: ignore[misc]
    )
    total_latency = sum(r.latency_ms for r in trace)

    total_tool_calls = sum(r.tool_calls for r in trace)
    total_tool_failures = sum(r.tool_failures for r in trace)
    tool_success_rate = (
        1.0 if total_tool_calls == 0 else 1.0 - total_tool_failures / total_tool_calls
    )

    retry_loops = sum(1 for r in trace if r.is_retry)
    correction_count = sum(1 for r in trace if r.is_correction)

    # Tokens/turns consumed up to and including the first useful answer.
    tokens_to_success: int | None = None
    turns_to_success: int | None = None
    if conv.success_turn is not None:
        upto = [r for r in trace if r.turn_index <= conv.success_turn]
        tokens_to_success = sum(r.input_tokens + r.output_tokens for r in upto)
        turns_to_success = sum(1 for r in upto if r.role == Role.ASSISTANT.value)

    convergence_efficiency = _convergence_efficiency(
        final_score=conv.final_score,
        converged=conv.converged,
        tokens_to_success=tokens_to_success,
        total_tokens=input_tokens + output_tokens,
        corrections=correction_count,
    )

    return AgentReport(
        agent=conv.agent,
        task_id=conv.task_id,
        model=conv.model,
        provider=conv.provider,
        turn_count=len(conv.turns),
        assistant_turns=len(conv.assistant_turns),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        usage_details=usage_details,
        provided_usage_details=provided_usage_details,
        tool_calls=total_tool_calls,
        tool_success_rate=round(tool_success_rate, 4),
        retry_loops=retry_loops,
        correction_count=correction_count,
        latency_estimate_ms=round(total_latency, 1),
        cost_estimate_usd=None if total_cost is None else round(total_cost, 6),
        cost_details={k: round(v, 6) for k, v in cost_details.items()},
        provided_cost_details={k: round(v, 6) for k, v in provided_cost_details.items()},
        success_turn=conv.success_turn,
        turns_to_success=turns_to_success,
        tokens_to_success=tokens_to_success,
        final_correct=conv.final_correct,
        final_score=conv.final_score,
        convergence_efficiency=convergence_efficiency,
        trace=trace,
    )


def _sum_int_maps(maps) -> dict[str, int]:
    total: dict[str, int] = {}
    for item in maps:
        for key, value in item.items():
            total[key] = total.get(key, 0) + int(value)
    return total


def _sum_float_maps(maps) -> dict[str, float]:
    total: dict[str, float] = {}
    for item in maps:
        for key, value in item.items():
            total[key] = total.get(key, 0.0) + float(value)
    return total


def _finalize_cost_details(details: dict[str, float]) -> dict[str, float]:
    """Ensure a cost map has ``total`` when it can be derived."""

    if not details or "total" in details:
        return details
    details["total"] = sum(details.values())
    return details


def _total_cost_from_details(details: dict[str, float]) -> float:
    return details["total"] if "total" in details else sum(details.values())


def _convergence_efficiency(
    *,
    final_score: float,
    converged: bool,
    tokens_to_success: int | None,
    total_tokens: int,
    corrections: int,
) -> float:
    """A 0-1 score rewarding correct answers reached with few tokens/corrections.

    Combines result quality with how much budget it took to get there. An agent
    that never converged scores 0; one that converged cheaply and without user
    corrections approaches 1.
    """

    if not converged or final_score <= 0:
        return 0.0

    denom = tokens_to_success or total_tokens or 1
    # Normalize against a reference budget of 4k tokens-to-success.
    budget_term = min(1.0, 4000.0 / denom)
    correction_penalty = 1.0 / (1.0 + corrections)
    return round(final_score * budget_term * correction_penalty, 4)


def analyze_many(
    conversations: list[Conversation], prices: PriceBook | None = None
) -> list[AgentReport]:
    prices = prices or PriceBook()
    return [analyze(c, prices) for c in conversations]
