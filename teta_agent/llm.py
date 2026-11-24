"""The synthesis-model seam.

WHY THIS MODULE EXISTS
----------------------
``rag_chat.py`` calls ``openai.OpenAI(...).chat.completions.create(...)``
inline. That is fine for one script, but it welds three separate decisions
together: *which* model, *where* it is hosted, and *whether a network is
available at all*. The orchestrator needs to run in CI, in a demo with no
tunnel up, and in production against a large model -- so those decisions have
to come apart.

A provider here is anything with ``.complete(system, user) -> str``. That is
the entire contract. Four implementations ship:

    EchoLLM        no network, deterministic   -> tests, offline demos
    OpenAICompatLLM  any OpenAI-shaped endpoint -> argo-proxy, vLLM, Genesis MAG
    NullLLM        raises if called            -> proves routing needs no LLM
    ScriptedLLM    replays canned turns        -> offline agent-loop demos

TWO CONTRACTS, NOT ONE
----------------------
``complete(system, user) -> str`` is the synthesis contract, used by the
deterministic ``Orchestrator``. It stays minimal on purpose: synthesis must
work against any endpoint, including ones with no tool-calling support.

``chat(messages, tools) -> LLMTurn`` is the *agent* contract, used by
``AgentOrchestrator``. It is optional. Providers that implement it can drive
tool selection; providers that do not simply lack the method, and
``supports_tools()`` reports which is which. Keeping them separate means
adding an agent loop could not break the deterministic path.

SWAPPING IN A LARGER MODEL
--------------------------
Change one line at the call site::

    orch = Orchestrator(llm=EchoLLM())                      # offline
    orch = Orchestrator(llm=OpenAICompatLLM())              # argo default
    orch = Orchestrator(llm=OpenAICompatLLM(model="argo:gpt-5"))
    orch = Orchestrator(llm=OpenAICompatLLM(
        base_url="https://api.i2-core.american-science-cloud.org/v1",
        model="llama-3.3-70b", api_key=os.environ["AMSC_I2_API_KEY"]))

No orchestrator code changes, because the orchestrator never imports
``openai`` -- see ``orchestrator.py``.

CPU NOTE
--------
Nothing in this module runs a model locally. ``EchoLLM`` is pure string
handling and ``OpenAICompatLLM`` is an HTTP client. All GPU cost, if any,
lives on the far side of ``base_url``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json

#: Defaults mirror ``rag_chat.py`` so both paths hit the same endpoint.
DEFAULT_BASE_URL = "http://localhost:44497/v1"
DEFAULT_MODEL = "argo:claude-4.6-opus"


class LLMUnavailable(RuntimeError):
    """Raised when synthesis is attempted but no model can be reached.

    Distinct from a generic error so the orchestrator can degrade to
    "tools ran, synthesis skipped" instead of losing the tool results.
    """


# ── Tool calling ────────────────────────────────────────────────
#
# A second, richer contract alongside ``complete()``. Providers that support
# it expose ``chat(messages, tools) -> LLMTurn``; providers that do not simply
# omit the method, and ``AgentOrchestrator`` checks with ``hasattr``.
#
# Why two contracts rather than replacing the first: ``complete()`` is what
# the deterministic orchestrator uses for synthesis, and that path must keep
# working with no model capable of tool calls. Nothing below is required for
# it.


@dataclass
class ToolCall:
    """One tool invocation requested by the model.

    Attributes:
        call_id: Provider-assigned id. Must be echoed back on the result
            message or the model cannot match request to response.
        name: Tool name the model asked for. NOT trusted -- the caller
            resolves it against its own registry and rejects unknowns.
        arguments: Parsed JSON arguments. Empty dict when the model emitted
            malformed JSON; ``raw_arguments`` keeps the original for the
            trace, because "the model produced unparseable arguments" is
            itself an auditable event.
        raw_arguments: Argument string exactly as the model emitted it.
    """

    call_id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""


@dataclass
class LLMTurn:
    """One assistant turn: prose, tool calls, or both.

    Attributes:
        text: Assistant prose. Populated on the final turn; often empty
            while the model is still calling tools.
        tool_calls: Calls the model wants executed before it continues.
        raw_message: Provider message dict, appended verbatim to the
            transcript. Round-tripping the provider's own structure avoids
            reconstructing it incorrectly.
    """

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_message: Optional[Dict[str, Any]] = None

    def wants_tools(self):
        return bool(self.tool_calls)


def parse_tool_calls(message):
    """Extract ``ToolCall`` objects from an OpenAI-shaped message.

    Malformed argument JSON yields an empty ``arguments`` dict rather than
    raising. A model that emits bad JSON should be told so and given another
    turn -- that is a recoverable conversation state, not a crash.
    """
    calls = []
    for raw in getattr(message, "tool_calls", None) or []:
        function = getattr(raw, "function", None)
        if function is None:
            continue
        raw_args = getattr(function, "arguments", "") or ""
        try:
            parsed = json.loads(raw_args) if raw_args.strip() else {}
        except (ValueError, TypeError):
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        calls.append(
            ToolCall(
                call_id=getattr(raw, "id", "") or "",
                name=getattr(function, "name", "") or "",
                arguments=parsed,
                raw_arguments=raw_args,
            )
        )
    return calls


class EchoLLM:
    """Offline provider: deterministic, no network, no weights.

    Returns a structured restatement of the prompt rather than pretending to
    reason. It exists so the orchestrator's wiring -- dispatch, merge, trace
    construction -- can be exercised end to end with nothing running.

    Deliberately does NOT fabricate clinical prose. A plausible-looking fake
    answer in a demo is how stubs get mistaken for real output.
    """

    name = "echo"

    def __init__(self, marker="[offline synthesis]"):
        self.marker = marker

    def complete(self, system, user):
        """Return a deterministic placeholder derived from the prompt."""
        del system  # Intentionally unused; kept for interface parity.
        lines = [ln.strip() for ln in user.splitlines() if ln.strip()]
        head = lines[0] if lines else "(empty prompt)"
        return (
            "{} no model was called.\n"
            "Prompt had {} non-empty lines; first line: {}".format(
                self.marker, len(lines), head[:160]
            )
        )


class NullLLM:
    """Provider that refuses to be called.

    Used to *prove* a code path needs no model. If routing ever starts
    depending on synthesis, tests using this fail immediately and loudly
    rather than quietly making a network call.
    """

    name = "null"

    def complete(self, system, user):
        del system, user
        raise LLMUnavailable(
            "NullLLM.complete() was called -- this path is supposed to be "
            "model-free. Pass a real provider if synthesis is genuinely needed."
        )

    def chat(self, messages, tools):
        """Also refuses, so agent-loop paths are covered by the same proof."""
        del messages, tools
        raise LLMUnavailable(
            "NullLLM.chat() was called -- this path is supposed to be "
            "model-free."
        )


class OpenAICompatLLM:
    """Any OpenAI-shaped chat endpoint: argo-proxy, vLLM, LiteLLM, MAG.

    ``openai`` is imported lazily inside ``complete`` so that importing this
    module -- and therefore the orchestrator -- never requires the SDK. That
    is what lets the offline path have zero third-party dependencies.

    Args:
        model: Model id. Defaults to the one ``rag_chat.py`` uses.
        base_url: Endpoint root. Defaults to the local argo-proxy tunnel.
        api_key: For argo-proxy this is your ARGO *username*, not a key --
            see ``rag_chat.resolve_credential``. Falls back to ``ARGO_USER``.
        timeout: Seconds before giving up.
    """

    def __init__(
        self,
        model=DEFAULT_MODEL,
        base_url=DEFAULT_BASE_URL,
        api_key=None,
        timeout=120.0,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self._client = None

    @property
    def name(self):
        return self.model

    def _resolve_key(self):
        if self.api_key:
            return self.api_key
        import os

        return os.environ.get("ARGO_USER", "").strip() or "unset"

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise LLMUnavailable(
                    "openai SDK not installed; use EchoLLM for offline runs"
                ) from exc
            self._client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self._resolve_key(),
                timeout=self.timeout,
            )
        return self._client

    def complete(self, system, user):
        """Call the endpoint, translating transport errors into one type.

        Raises:
            LLMUnavailable: On any connection or auth failure. The caller
                gets one exception type to handle, and tool results survive.
        """
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            # argo-proxy reports a bad ARGO username as 403/auth, which reads
            # as "wrong API key" and sends you hunting for one that does not
            # exist. Name the real cause once, here.
            raise LLMUnavailable(
                "{} at {} failed: {}".format(
                    self.model, self.base_url, exc
                )
            ) from exc
        return response.choices[0].message.content or ""

    def chat(self, messages, tools):
        """One tool-calling turn.

        Args:
            messages: Full transcript so far, OpenAI message dicts. Sent
                whole every turn -- the endpoint is stateless, and the
                transcript IS the agent's memory.
            tools: JSON schemas in OpenAI ``{"type": "function", ...}`` form.

        Returns:
            LLMTurn.

        Raises:
            LLMUnavailable: On transport, auth, or unsupported-feature
                failure. Endpoints that reject the ``tools`` parameter land
                here too, which is the signal to fall back to the
                deterministic orchestrator.
        """
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or None,
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            raise LLMUnavailable(
                "{} at {} failed during tool-calling turn: {}".format(
                    self.model, self.base_url, exc
                )
            ) from exc

        message = response.choices[0].message
        calls = parse_tool_calls(message)

        # Rebuild the assistant message as a plain dict. The SDK object is
        # not JSON-serialisable and must go back into ``messages`` next turn.
        raw = {"role": "assistant", "content": message.content or ""}
        if calls:
            raw["tool_calls"] = [
                {
                    "id": c.call_id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.raw_arguments},
                }
                for c in calls
            ]
        return LLMTurn(
            text=message.content or "", tool_calls=calls, raw_message=raw
        )


class ScriptedLLM:
    """Offline provider that replays a fixed sequence of turns.

    Exists so the agent loop -- dispatch, trace construction, escalation,
    the guard that rejects unknown tools -- can be exercised with no network
    and no weights. The script is the *model's* behaviour, written down.

    This is a test double, not a simulation. It does not decide anything;
    pointing it at a demo and calling the result "the agent reasoning" would
    be exactly the stub-mistaken-for-real failure the pathology module warns
    about.

    Args:
        turns: Sequence consumed in order. An entry may be an ``LLMTurn``,
            or a callable ``f(messages) -> LLMTurn`` when the turn needs to
            depend on what the tools actually returned. Running past the end
            yields a final turn with no tool calls, so a loop always
            terminates.

    On callable turns: a scripted agent that must react to a tool result
    cannot be a fixed list, because the result is not known when the script
    is written. Callables keep the script honest -- it reads the real
    observation instead of asserting a value the tool may not have produced.
    This is string handling, not inference.
    """

    name = "scripted"

    def __init__(self, turns=None):
        self.turns = list(turns or [])
        self.calls_made = 0

    def complete(self, system, user):
        del system, user
        return "[scripted] no model was called."

    def chat(self, messages, tools):
        del tools
        index = self.calls_made
        self.calls_made += 1
        if index < len(self.turns):
            entry = self.turns[index]
            return entry(messages) if callable(entry) else entry
        return LLMTurn(
            text="[scripted] script exhausted; ending turn.",
            raw_message={
                "role": "assistant",
                "content": "[scripted] script exhausted; ending turn.",
            },
        )


def supports_tools(llm):
    """Whether ``llm`` can drive an agent loop."""
    return callable(getattr(llm, "chat", None))


def default_llm(offline=False):
    """Pick a provider. ``offline=True`` guarantees no network."""
    return EchoLLM() if offline else OpenAICompatLLM()
