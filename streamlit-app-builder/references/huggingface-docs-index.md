# HuggingFace Docs Index

Canonical `huggingface.co/docs` URLs used by the `streamlit-app-builder` skill. Fetch these at skill-run time to verify current HF library APIs before generating code.

## Baseline

Fetched whenever the generated code will use `huggingface_hub`, `transformers`, or `diffusers` (i.e., any HF-card input, and any code/notebook input that imports one of those libraries).

| Topic                            | URL                                                              |
|----------------------------------|------------------------------------------------------------------|
| Hub security tokens (gated auth) | https://huggingface.co/docs/hub/en/security-tokens               |
| `huggingface-cli` login          | https://huggingface.co/docs/huggingface_hub/en/guides/cli        |

## Library-conditional

Fetch a row when its trigger matches the input.

| Trigger                                                                                       | Topic                                | URL                                                                |
|-----------------------------------------------------------------------------------------------|--------------------------------------|--------------------------------------------------------------------|
| `library_name == "transformers"` or code input imports `transformers`                          | Pipelines (signatures, task mapping) | https://huggingface.co/docs/transformers/en/main_classes/pipelines |
| `library_name == "transformers"` or code input imports `transformers`                          | Auto classes                         | https://huggingface.co/docs/transformers/en/model_doc/auto         |
| `library_name == "transformers"` AND `pipeline_tag == "automatic-speech-recognition"`          | ASR task guide                       | https://huggingface.co/docs/transformers/en/tasks/asr              |
| `library_name == "transformers"` AND `pipeline_tag == "text-to-speech"`                        | TTS task guide                       | https://huggingface.co/docs/transformers/en/tasks/text-to-speech   |
| `library_name == "diffusers"` or code input imports `diffusers`                                | Loading pipelines                    | https://huggingface.co/docs/diffusers/en/using-diffusers/loading   |
| `library_name == "diffusers"` or code input imports `diffusers`                                | Quick tour                           | https://huggingface.co/docs/diffusers/en/quicktour                 |

## Root

| Area         | URL                          |
|--------------|------------------------------|
| HF docs root | https://huggingface.co/docs  |

## Out of scope

- `sentence-transformers` docs live at `sbert.net` (off-domain).
- `mlx-*` library docs live on GitHub (`ml-explore/mlx`, off-domain).
- Both are intentionally excluded to keep this index within `huggingface.co/docs`. The skill declares the relevant PyPI deps in Step 6 without needing doc references here.
