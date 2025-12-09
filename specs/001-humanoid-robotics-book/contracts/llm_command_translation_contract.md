# Contract: LLM Natural Language Command Translation

## Purpose
To translate natural language commands into a structured format for robotic actions.

## Interface
- LLM API Call (e.g., OpenAI API)

## Inputs
- `natural_language_command`: (string) User's command (e.g., "Pick up the red cube").

## Outputs
- `action_type`: (string) Categorized action (e.g., "grasp", "navigate", "speak").
- `parameters`: (JSON object) Key-value pairs defining action specifics (e.g., `{"object": "red cube", "location": "table"}`).
- `confidence`: (float) Confidence score of the translation.

## Error Taxonomy
- `UNKNOWN_COMMAND`: The command could not be understood or mapped to a known action.
- `MISSING_PARAMETERS`: Essential parameters for the action were not identified.
- `LOW_CONFIDENCE`: The translation confidence is below a predefined threshold.
