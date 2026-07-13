"""
Optional auto-instrumentation for popular LLM client libraries.

``swarmtrace.init()`` (default ``auto_instrument=True``) calls :func:`patch_all`,
which patches whichever of OpenAI, Anthropic, Gemini, and LiteLLM are
installed, so raw LLM calls are traced as ``kind="llm"`` — attributed to
whatever ``@observe``'d agent is currently running, or to themselves if
none — with zero decorators at the call site.

Production guarantees
---------------------
- **Non-blocking**: trace recording is enqueued onto the background sender
  thread, never on the calling thread. The LLM call's latency is unaffected.
- **Idempotent**: each client method is only wrapped once — safe to call
  ``patch_all()`` (or ``swarmtrace.init()``) multiple times.
- **Exception-transparent**: the original exception always propagates to the
  caller; the trace records the error string but never swallows or delays it.
- **No content capture**: only metadata is recorded (model, latency, tokens,
  cost). Prompt/response content is never persisted by auto-instrumentation.
- **fov-compatible**: checks ``__swarmtrace_patched__`` before wrapping, so the
  fov stream patches (which also wrap OpenAI) don't produce double traces.

Adding a provider
------------------
Each ``patch_<provider>()`` function below wires one or two client methods
through the shared ``_wrap_call`` / ``_wrap_acall`` control flow (the
try/except/finally + stream-vs-non-stream dispatch is written once, not once
per provider). What differs per provider is small and declared inline:
which method to patch, how to read the model name off the call, and which
attribute/field names hold token usage on the response. See any existing
``patch_*`` function below for the pattern — a new OpenAI-compatible provider
is typically ~10 lines.
"""

import functools
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple

from swarmtrace.pricing import calculate_cost
from swarmtrace.redact import redact
import swarmtrace.tracer as _tracer

_log = logging.getLogger("swarmtrace")


# ---------------------------------------------------------------------------
# Stream wrappers — defer trace recording until the stream is exhausted
# ---------------------------------------------------------------------------
# When stream=True, the LLM client returns a generator/iterator, NOT a
# response object. The old code read response.usage immediately (didn't
# exist → 0 tokens) and recorded the trace in the finally block before any
# chunks were consumed (latency ≈ 0). These wrappers intercept the stream,
# accumulate usage metadata from chunks, and only call _record_async when
# the stream is fully exhausted or breaks.

def _accumulate_stream_usage(chunk, in_tok: int, out_tok: int, model: str):
    """Extract/accumulate usage + model from one stream chunk, across the
    shapes different providers use. Shared by both wrapper classes below
    (previously duplicated verbatim in each).

      - OpenAI / LiteLLM: chunk.usage on the final chunk (if stream_options
        includes include_usage)
      - Anthropic: message_start.event.usage.input_tokens,
        message_delta.usage.output_tokens
      - LiteLLM: depends on underlying provider

    We check all known shapes and keep the last non-zero value."""
    usage = getattr(chunk, "usage", None)
    if usage:
        in_tok = getattr(usage, "prompt_tokens", 0) or in_tok
        out_tok = getattr(usage, "completion_tokens", 0) or out_tok
        # Anthropic-style: input_tokens / output_tokens
        in_tok = getattr(usage, "input_tokens", 0) or in_tok
        out_tok = getattr(usage, "output_tokens", 0) or out_tok
    m = getattr(chunk, "model", None)
    if m:
        model = m
    # Anthropic streaming events have a .type attribute
    chunk_type = getattr(chunk, "type", None)
    if chunk_type == "message_start":
        msg = getattr(chunk, "message", None)
        if msg:
            u = getattr(msg, "usage", None)
            if u:
                in_tok = getattr(u, "input_tokens", 0) or in_tok
    elif chunk_type == "message_delta":
        u = getattr(chunk, "usage", None)
        if u:
            out_tok = getattr(u, "output_tokens", 0) or out_tok
    return in_tok, out_tok, model


class _StreamInstrumentWrapper:
    """Wraps a sync streaming response. Records the trace when the stream
    is exhausted or raises.

    Implements __enter__/__exit__/__getattr__ so it works as a context
    manager (`with client.chat.completions.create(..., stream=True) as s:`)
    and so attribute access on the underlying stream (e.g. .response, .parse())
    still works — OpenAI's stream objects support both patterns."""

    def __init__(self, stream, func_name, model, start, agent, parent_id):
        self._stream = stream
        self._func_name = func_name
        self._model = model
        self._start = start
        self._agent = agent
        self._parent_id = parent_id
        self._in_tok = 0
        self._out_tok = 0
        self._error: Optional[Exception] = None
        self._recorded = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
            self._extract_usage(chunk)
            return chunk
        except StopIteration:
            self._record()
            raise
        except Exception as exc:
            self._error = exc
            self._record()
            raise

    def __enter__(self):
        # Support `with ... as stream:` — OpenAI streams are context managers.
        # Don't call __enter__ on the underlying stream; it may not have one.
        # The stream is already "entered" by the time we wrap it.
        return self

    def __exit__(self, *exc):
        # If the user exits the context manager without exhausting the stream,
        # record the trace with whatever we have so far (possibly 0 tokens).
        # This matches the non-streaming behavior where the finally block
        # always records.
        self._record()
        return False  # don't suppress exceptions

    def __getattr__(self, name):
        # Passthrough for attribute access on the underlying stream
        # (e.g. stream.response, stream.parse(), stream.close()).
        # Only called when normal attribute lookup fails on self.
        # Guard against infinite recursion if _stream isn't set yet
        # (e.g. during __init__ or unpickling) — raise AttributeError
        # rather than recursing into __getattr__ for '_stream'.
        if name == "_stream":
            raise AttributeError(name)
        return getattr(self._stream, name)

    def _extract_usage(self, chunk):
        self._in_tok, self._out_tok, self._model = _accumulate_stream_usage(
            chunk, self._in_tok, self._out_tok, self._model
        )

    def _record(self):
        if self._recorded:
            return
        self._recorded = True
        _record_async(self._func_name, self._model, self._start,
                      self._error, self._in_tok, self._out_tok,
                      self._agent, self._parent_id)


class _AsyncStreamInstrumentWrapper:
    """Wraps an async streaming response. Records the trace when the stream
    is exhausted or raises.

    Implements __aenter__/__aexit__/__getattr__ so it works as an async
    context manager (`async with client.chat.completions.create(...,
    stream=True) as s:`) and so attribute access on the underlying stream
    still works."""

    def __init__(self, stream, func_name, model, start, agent, parent_id):
        self._stream = stream
        self._func_name = func_name
        self._model = model
        self._start = start
        self._agent = agent
        self._parent_id = parent_id
        self._in_tok = 0
        self._out_tok = 0
        self._error: Optional[Exception] = None
        self._recorded = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()
            self._extract_usage(chunk)
            return chunk
        except StopAsyncIteration:
            self._record()
            raise
        except Exception as exc:
            self._error = exc
            self._record()
            raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        # Record the trace if the user exits without exhausting the stream.
        self._record()
        return False  # don't suppress exceptions

    def __getattr__(self, name):
        # Passthrough for attribute access on the underlying stream.
        # Note: async methods on the underlying stream will be returned as
        # regular functions; the caller must await them. This matches how
        # OpenAI's async stream objects expose methods like .parse().
        # Guard against infinite recursion if _stream isn't set yet.
        if name == "_stream":
            raise AttributeError(name)
        return getattr(self._stream, name)

    def _extract_usage(self, chunk):
        self._in_tok, self._out_tok, self._model = _accumulate_stream_usage(
            chunk, self._in_tok, self._out_tok, self._model
        )

    def _record(self):
        if self._recorded:
            return
        self._recorded = True
        _record_async(self._func_name, self._model, self._start,
                      self._error, self._in_tok, self._out_tok,
                      self._agent, self._parent_id)


def _record_async(
    func_name: str,
    model: str,
    start: float,
    error: Optional[Exception],
    in_tok: int,
    out_tok: int,
    agent: Optional[Tuple[str, str]],
    parent_id: Optional[str],
) -> None:
    """Fire-and-forget: build the trace record and hand it to the background
    sender. Called in a ``finally`` block so it must never raise.
    """
    try:
        cost = calculate_cost(model or "", in_tok, out_tok)
        trace_id = _tracer._build_trace_id()
        agent_id, agent_name = agent or (trace_id, func_name)
        timestamp = datetime.now(timezone.utc).isoformat()
        latency = round(time.perf_counter() - start, 3)
        # Redact the error string — LLM auth errors (esp. older OpenAI
        # clients, some Anthropic error shapes) can echo the API key back
        # in the exception message. This is the exact PII leak that
        # swarmtrace/redact.py was built to catch, but the original
        # Task 1 commit missed this path because the args_str/output
        # strings are synthesized ("model=…") and don't carry user
        # content. The error string DOES — it comes from the provider's
        # exception, which we don't control.
        error_str = redact(str(error)) if error else None
        output = None if error else f"model={model} tokens={in_tok}in/{out_tok}out"
        args_str = f"model={model}"
        # save_trace writes to SQLite — fast local I/O, exception-safe.
        # Using module reference so tests can monkeypatch tracer.save_trace.
        session_id = _tracer._current_session()
        _tracer.save_trace(
            id_=trace_id, parent_id=parent_id, function=func_name,
            args=args_str, output=output, latency_sec=latency, error=error_str,
            timestamp=timestamp, input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=cost, kind="llm", agent_id=agent_id, agent_name=agent_name,
            session_id=session_id,
        )
        _tracer._enqueue_remote({
            "id": trace_id, "parent_id": parent_id, "function": func_name,
            "args": args_str, "output": output or "", "latency_sec": latency,
            "error": error_str, "timestamp": timestamp,
            "input_tokens": in_tok, "output_tokens": out_tok, "cost_usd": cost,
            "kind": "llm", "agent_id": agent_id, "agent_name": agent_name,
        })
    except Exception as exc:
        _log.warning("auto-instrument record warning: %s", exc)


def _already_patched(target) -> bool:
    return getattr(target, "__swarmtrace_patched__", False)


def _mark_patched(wrapper):
    wrapper.__swarmtrace_patched__ = True
    return wrapper


# ---------------------------------------------------------------------------
# Shared patch control flow
# ---------------------------------------------------------------------------
# Every provider's create/complete method needs the same wrapping logic:
# start a timer, look up the current agent/parent, call the original, and
# either defer to a stream wrapper (stream=True) or read usage off the
# response and record immediately — all while letting the original
# exception propagate untouched. This used to be copy-pasted once per sync
# method and once per async method (8 near-identical copies across 4
# providers). `_wrap_call` / `_wrap_acall` write it once; each provider
# below only supplies what's actually different: how to read the model
# name off the call, and how to read usage off the response.

ModelFn = Callable[[object, tuple, dict], str]
UsageFn = Callable[[object], Tuple[int, int, Optional[str]]]


def _model_from_kwargs(self_obj, args, kwargs) -> str:
    """Default model extractor: OpenAI/Anthropic-style clients always pass
    model= as a kwarg."""
    return kwargs.get("model", "")


def _usage_from(
    usage_attr: str = "usage",
    in_field: str = "prompt_tokens",
    out_field: str = "completion_tokens",
    model_attr: Optional[str] = "model",
) -> UsageFn:
    """Build a usage extractor for a non-streaming response. Covers every
    provider here: they all expose token counts as two fields nested under
    one attribute on the response — only the names differ.
      - OpenAI / LiteLLM: response.usage.{prompt_tokens,completion_tokens}
      - Anthropic:        response.usage.{input_tokens,output_tokens}
      - Gemini:            response.usage_metadata.{prompt_token_count,
                            candidates_token_count} (no model on response)
    """
    def extractor(response) -> Tuple[int, int, Optional[str]]:
        usage = getattr(response, usage_attr, None)
        in_tok = getattr(usage, in_field, 0) or 0
        out_tok = getattr(usage, out_field, 0) or 0
        model = getattr(response, model_attr, None) if model_attr else None
        return in_tok, out_tok, model
    return extractor


def _wrap_call(original, op_name: str, model_fn: ModelFn, usage_fn: UsageFn, *, bound: bool):
    """Wrap a sync provider method/function with the shared trace-recording
    control flow. ``bound=True`` means the original is called as an
    instance method (first positional arg is ``self``, as with
    Completions.create); ``bound=False`` means it's a plain function (as
    with litellm.completion)."""
    @functools.wraps(original)
    def patched(*call_args, **kwargs):
        start = time.perf_counter()
        self_obj = call_args[0] if bound else None
        args = call_args[1:] if bound else call_args
        model = model_fn(self_obj, args, kwargs)
        agent = _tracer._current_agent()
        parent_id = _tracer._current_parent()
        error: Optional[Exception] = None
        in_tok = out_tok = 0
        is_stream = kwargs.get("stream", False)
        # stream_returned tracks whether original() successfully returned
        # a stream (vs raised). If it raised, we must record the error
        # trace here in the finally block — there's no stream wrapper to
        # defer to. If it returned a stream, the wrapper handles recording.
        stream_returned = False
        try:
            response = original(*call_args, **kwargs)
            if is_stream:
                stream_returned = True
                return _StreamInstrumentWrapper(
                    response, op_name, model, start, agent, parent_id,
                )
            in_tok, out_tok, model_override = usage_fn(response)
            model = model_override or model
            return response
        except Exception as exc:
            error = exc
            raise
        finally:
            # Record here ONLY if the stream wasn't returned (either
            # non-stream, or stream that raised before returning).
            # If the stream was returned, the wrapper records on exhaustion.
            if not stream_returned:
                _record_async(op_name, model, start, error, in_tok, out_tok, agent, parent_id)

    return patched


def _wrap_acall(original, op_name: str, model_fn: ModelFn, usage_fn: UsageFn, *, bound: bool):
    """Async counterpart to _wrap_call — same control flow, awaited."""
    @functools.wraps(original)
    async def patched(*call_args, **kwargs):
        start = time.perf_counter()
        self_obj = call_args[0] if bound else None
        args = call_args[1:] if bound else call_args
        model = model_fn(self_obj, args, kwargs)
        agent = _tracer._current_agent()
        parent_id = _tracer._current_parent()
        error: Optional[Exception] = None
        in_tok = out_tok = 0
        is_stream = kwargs.get("stream", False)
        stream_returned = False
        try:
            response = await original(*call_args, **kwargs)
            if is_stream:
                stream_returned = True
                return _AsyncStreamInstrumentWrapper(
                    response, op_name, model, start, agent, parent_id,
                )
            in_tok, out_tok, model_override = usage_fn(response)
            model = model_override or model
            return response
        except Exception as exc:
            error = exc
            raise
        finally:
            if not stream_returned:
                _record_async(op_name, model, start, error, in_tok, out_tok, agent, parent_id)

    return patched


# ---------------------------------------------------------------------------
# OpenAI (and OpenAI-compatible: Mistral, DeepSeek, Groq, Together, …)
# ---------------------------------------------------------------------------

def patch_openai() -> bool:
    try:
        from openai.resources.chat.completions import Completions, AsyncCompletions
    except ImportError:
        return False

    op_name = "openai.chat.completions.create"
    usage_fn = _usage_from()  # response.usage.{prompt_tokens,completion_tokens}

    if not _already_patched(Completions.create):
        Completions.create = _mark_patched(
            _wrap_call(Completions.create, op_name, _model_from_kwargs, usage_fn, bound=True)
        )

    if not _already_patched(AsyncCompletions.create):
        AsyncCompletions.create = _mark_patched(
            _wrap_acall(AsyncCompletions.create, op_name, _model_from_kwargs, usage_fn, bound=True)
        )

    return True


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def patch_anthropic() -> bool:
    try:
        from anthropic.resources.messages import Messages, AsyncMessages
    except ImportError:
        return False

    op_name = "anthropic.messages.create"
    usage_fn = _usage_from(in_field="input_tokens", out_field="output_tokens")

    if not _already_patched(Messages.create):
        Messages.create = _mark_patched(
            _wrap_call(Messages.create, op_name, _model_from_kwargs, usage_fn, bound=True)
        )

    if not _already_patched(AsyncMessages.create):
        AsyncMessages.create = _mark_patched(
            _wrap_acall(AsyncMessages.create, op_name, _model_from_kwargs, usage_fn, bound=True)
        )

    return True


# ---------------------------------------------------------------------------
# Google Gemini (google-generativeai)
# ---------------------------------------------------------------------------

def patch_gemini() -> bool:
    try:
        from google.generativeai import GenerativeModel
    except ImportError:
        return False

    def _model_name(self) -> str:
        name = getattr(self, "model_name", "") or getattr(self, "_model_name", "") or ""
        return name.removeprefix("models/")

    op_name = "gemini.generate_content"
    # Gemini's response doesn't carry the model back, unlike the others —
    # model_attr=None means usage_fn never overrides the model we passed in.
    usage_fn = _usage_from(
        usage_attr="usage_metadata",
        in_field="prompt_token_count",
        out_field="candidates_token_count",
        model_attr=None,
    )

    def model_fn(self_obj, args, kwargs) -> str:
        return _model_name(self_obj)

    if not _already_patched(GenerativeModel.generate_content):
        GenerativeModel.generate_content = _mark_patched(
            _wrap_call(GenerativeModel.generate_content, op_name, model_fn, usage_fn, bound=True)
        )

    original_async = getattr(GenerativeModel, "generate_content_async", None)
    if original_async is not None and not _already_patched(original_async):
        GenerativeModel.generate_content_async = _mark_patched(
            _wrap_acall(original_async, op_name, model_fn, usage_fn, bound=True)
        )

    return True


# ---------------------------------------------------------------------------
# LiteLLM (covers Mistral, DeepSeek, Cohere, Bedrock, Azure, … via one SDK)
# ---------------------------------------------------------------------------

def patch_litellm() -> bool:
    try:
        import litellm
    except ImportError:
        return False

    op_name = "litellm.completion"
    usage_fn = _usage_from()  # same shape as OpenAI
    # litellm.completion(*args, **kwargs) is a plain function, not a bound
    # method — model can arrive positionally (args[0]) or as a kwarg.
    def model_fn(self_obj, args, kwargs) -> str:
        return kwargs.get("model") or (args[0] if args else "")

    if not _already_patched(litellm.completion):
        litellm.completion = _mark_patched(
            _wrap_call(litellm.completion, op_name, model_fn, usage_fn, bound=False)
        )

    if not _already_patched(litellm.acompletion):
        litellm.acompletion = _mark_patched(
            _wrap_acall(litellm.acompletion, op_name, model_fn, usage_fn, bound=False)
        )

    return True


def patch_all() -> dict:
    """Patch every supported LLM client that's installed. Safe to call repeatedly.

    Returns which clients are active, e.g.::

        {"openai": True, "anthropic": False, "gemini": False, "litellm": True}

    A client being ``False`` just means that SDK isn't installed in this
    environment — everything else keeps tracing normally. Also printed to
    stderr as one line, the same way ``fov.patch_all()`` reports its own
    active patches, so ``init()`` never leaves you guessing which LLM calls
    are actually being traced.
    """
    patches = {
        "openai":    patch_openai,
        "anthropic": patch_anthropic,
        "gemini":    patch_gemini,
        "litellm":   patch_litellm,
    }
    results: dict = {}
    for name, patch in patches.items():
        try:
            results[name] = bool(patch())
        except Exception as exc:
            results[name] = False
            _log.warning("auto-instrument warning (%s): %s", patch.__name__, exc)
    active = [name for name, ok in results.items() if ok]
    _log.info("llm auto-instrument active: %s", ', '.join(active) or 'none installed')
    return results
