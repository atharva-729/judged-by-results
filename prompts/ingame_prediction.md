You label Reddit comments posted **during** 2025 World Series Game 1, Los Angeles Dodgers (LAD) at Toronto Blue Jays (TOR). Each comment says which fanbase posted it: LAD fan, TOR fan, or neutral. "We" means the writer's own team.

For each comment decide whether it is a **prediction about who will win this game** (not the series, not a single at-bat). Examples: "this game is over", "we're coming back", "we've got this", "no way they lose this now".

Return only a JSON array, one object per comment, in the same order:

```json
[{"id": "<id>", "is_prediction": true, "predicted_winner": "LAD" | "TOR" | null, "confidence": "low" | "mid" | "high"}]
```

- `predicted_winner` is null when `is_prediction` is false.
- `confidence` is how sure the writer sounds ("might come back" = low; "this is over" = high).

Comments:

{{comments}}
