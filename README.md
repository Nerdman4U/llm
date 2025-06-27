# LLM

Wrapper classes for Hugging Face Transformers LLM models.

NOTE: Package has a dependency of not-yet published package! To make it work without, must
remove InitializationParams and just read arguments without it.

## How to use

Initialize virtual environment.
`$ python -m venv .venv`

Set up environment, aliases and install packages.
`$ source .init_workspace`

Run tests.
`$ pyt`

Try out transformer classes.
`$ python src/llm/extension/t5_transformer.py`

And finally import them at your code! \o/ XD

## Environment

To use Google CodeGemma models, set the `HUGGING_FACE_TOKEN` environment variable with your Hugging Face access token (You need to signup at their page).

## Cache

By default models are cached to ./cache.

## Tests

Only generated tests exists currently.
