"""Task removals: official autocomplete semantics and verified catalog errata."""

from collections.abc import Mapping


def hidden_columns(
    extra: Mapping[str, object],
    *,
    name: str,
    entity_table: str | None,
    target_column: str | None,
) -> tuple[tuple[str, str], ...]:
    raw = extra.get("hidden_columns", [])
    if raw is None:
        raw = []
    if not isinstance(raw, (list, tuple)):
        raise TypeError("hidden_columns must be a list of [table, column] pairs")
    pairs: list[tuple[str, str]] = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise TypeError("hidden_columns must contain [table, column] pairs")
        pairs.append((str(pair[0]), str(pair[1])))
    # The hosted external Ratebeer manifest omits its directly copied label.
    # This exact task erratum must never hide same-name historical forecast columns.
    ratebeer_label = (
        name == "beer_ratings-total_score"
        and entity_table == "beer_ratings"
        and target_column == "total_score"
        and extra.get("kind") == "external"
    )
    if extra.get("kind") == "autocomplete" or ratebeer_label:
        if not entity_table or not target_column:
            raise ValueError("A column prediction task requires entity and target")
        pairs.append((entity_table, target_column))
    return tuple(dict.fromkeys(pairs))
