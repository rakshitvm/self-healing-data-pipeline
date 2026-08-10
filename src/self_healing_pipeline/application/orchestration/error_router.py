"""Tier 1 error router.

Routes a `PipelineError` instance to a registered `RepairAgent` using
inheritance-aware dispatch: `resolve` walks the error's method resolution
order (MRO), most specific type first, and returns the handler registered
against the closest matching ancestor. There is no `if`/`elif` chain and
no hardcoded exception-to-agent mapping anywhere in this module — routing
is entirely data-driven through `register`, so a new `PipelineError`
subclass (Tier 2, Tier 3, or otherwise) is wired up purely by registering
a handler for it. The router itself never needs to change, and it never
imports a concrete repair agent — only the `RepairAgent` abstraction.
"""

from self_healing_pipeline.domain.exceptions.domain_exceptions import PipelineError
from self_healing_pipeline.domain.interfaces.agents.repair_agent import RepairAgent
from self_healing_pipeline.domain.value_objects.repair_result import RepairResult


class NoHandlerRegisteredError(Exception):
    """Raised when an error has no exact, inherited, or fallback handler."""

    def __init__(self, error_type: type[PipelineError]) -> None:
        super().__init__(
            f"No handler registered for {error_type.__name__!r} and no "
            "fallback handler is registered."
        )
        self.error_type = error_type


class ErrorRouter:
    """Registry-based, inheritance-aware router from errors to repair agents.

    Handlers are registered against `PipelineError` subclasses via
    `register`. Resolving an error instance searches its MRO — most
    specific type first — for the closest registered ancestor, so an
    exact registration always outranks a registration on a parent class,
    and a nearer ancestor always outranks a farther one. Registering a
    handler for `PipelineError` itself acts as a catch-all, since
    `PipelineError` is the last domain type in every error's MRO. If no
    type in the MRO has a registered handler, the registered fallback
    handler (if any) is used instead.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[PipelineError], RepairAgent] = {}
        self._fallback: RepairAgent | None = None

    def register(self, error_type: type[PipelineError], handler: RepairAgent) -> None:
        """Register `handler` to be used for `error_type` and its subclasses.

        A more specific registration (e.g. on a subclass of `error_type`)
        always takes priority over this one when routing an instance of
        that subclass.
        """
        self._handlers[error_type] = handler

    def register_fallback(self, handler: RepairAgent) -> None:
        """Register the handler used when no type in the MRO has a match."""
        self._fallback = handler

    def resolve(self, error: PipelineError) -> RepairAgent:
        """Return the handler that routing `error` would dispatch to.

        Behavior:
        - Exact match: `type(error)` has a registered handler -> used.
        - Inherited match: no exact match, but a registered handler exists
          on the closest ancestor in `type(error).__mro__` -> used.
        - Fallback match: no type in the MRO has a registered handler, but
          a fallback handler is registered -> used.
        - No handler and no fallback: raises `NoHandlerRegisteredError`.
        """
        for candidate_type in type(error).__mro__:
            if candidate_type in self._handlers:
                return self._handlers[candidate_type]
        if self._fallback is not None:
            return self._fallback
        raise NoHandlerRegisteredError(type(error))

    def route(self, error: PipelineError) -> RepairResult:
        """Resolve a handler for `error` and delegate the repair to it."""
        handler = self.resolve(error)
        return handler.handle(error)
