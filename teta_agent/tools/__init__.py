"""Agent-callable tools.

Each module here exposes one *capability* the orchestrator can invoke. Tools
are deliberately uniform in shape so the orchestrator can treat them
interchangeably:

    - a module-level entry point that takes plain arguments,
    - a frozen result dataclass,
    - a ``to_trace_detail()`` on that result so the reasoning trace can record
      the call without knowing anything about the tool's domain.

``pathology`` is a STUB standing in for the TET histological subtype
classifier (Sacco et al., *Annals of Oncology* 2026). It matches the real
model's inputs and outputs; it does not look at pixels. See its docstring.
"""
