from typing import Any, Callable

from h2o_wave import Q
from h2o_wave.core import expando_to_dict
from h2o_wave.routing import (
    _arg_handlers,
    _event_handlers,
    _invoke_handler,
    _path_handlers,
)


async def _match_predicate(
    predicate: Callable, func: Callable, arity: int, q: Q, arg: Any
) -> bool:
    if predicate:
        if predicate(q, arg):
            await _invoke_handler(func, arity, q, arg)
            return True
    else:
        if arg:
            await _invoke_handler(func, arity, q, arg)
            return True
    return False


async def handle_on(q: Q) -> bool:
    """
    Handle the query using a query handler (a function annotated with `@on()`).

    Args:
        q: The query context.

    Returns:
        True if a matching query handler was found and invoked, else False.
    """
    event_sources = expando_to_dict(q.events)
    for event_source in event_sources:
        event = q.events[event_source]
        entries = _event_handlers.get(event_source)
        if entries:
            for entry in entries:
                event_type, predicate, func, arity = entry
                if event_type in event:
                    arg_value = event[event_type]
                    if await _match_predicate(predicate, func, arity, q, arg_value):
                        return True

    args = expando_to_dict(q.args)
    for arg in args:
        arg_value = q.args[arg]
        if arg == '#':
            for rx, conv, func, arity in _path_handlers:
                match = rx.match(arg_value)
                if match:
                    params = match.groupdict()
                    for key, value in params.items():
                        params[key] = conv[key].convert(value)
                    if len(params):
                        if arity <= 1:
                            await _invoke_handler(func, arity, q, None)  # type: ignore
                        else:
                            await func(q, **params)
                    else:
                        await _invoke_handler(func, arity, q, None)  # type: ignore
                    return True
        else:
            entries = _arg_handlers.get(arg)
            if entries:
                for entry in entries:
                    predicate, func, arity = entry
                    if await _match_predicate(predicate, func, arity, q, arg_value):
                        return True
    return False
