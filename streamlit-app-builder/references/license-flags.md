# License Flags

Licenses that the `streamlit-app-builder` skill flags in generated `README.md` under "License & Commercial Use". When the model card's `license` field matches (or `license_name` for custom licenses), surface the flag prominently — teams deploying to paying customers need to know commercial restrictions upfront.

## Licenses that restrict commercial use

| License identifier / name         | Commercial use                                                   | Notes                                                                 |
|-----------------------------------|------------------------------------------------------------------|-----------------------------------------------------------------------|
| `cc-by-nc-2.0`, `-3.0`, `-4.0`    | ❌ Not allowed                                                    | Non-commercial only                                                   |
| `cc-by-nc-sa-*`                   | ❌ Not allowed                                                    | Non-commercial + share-alike                                          |
| `cc-by-nc-nd-*`                   | ❌ Not allowed                                                    | Non-commercial + no-derivatives                                       |
| `llama2`                          | ⚠️ Restricted                                                    | Llama 2 Community License — free under 700M MAU threshold             |
| `llama3`                          | ⚠️ Restricted                                                    | Llama Community License — 700M MAU threshold                          |
| `llama3.1`                        | ⚠️ Restricted                                                    | Llama Community License — 700M MAU threshold                          |
| `llama3.2`                        | ⚠️ Restricted                                                    | Llama Community License — 700M MAU threshold                          |
| `llama3.3`                        | ⚠️ Restricted                                                    | Llama Community License — 700M MAU threshold                          |
| `gemma`                           | ⚠️ Restricted                                                    | Gemma Terms of Use — use-case restrictions apply                      |
| `other` with `license_name: mistral-ai-research-license` | ❌ Research-only                           | Mistral Research License                                              |
| `openrail`, `bigscience-openrail-m`, `creativeml-openrail-m` | ⚠️ Use-case restricted                | RAIL licenses — enumerated prohibited uses                            |
| `deepfloyd-if-license`            | ⚠️ Research-only by default                                      | Commercial permitted after separate application                       |
| `apple-ascl`                      | ⚠️ Restricted                                                    | Apple Sample Code License — internal/research                         |

## Licenses that permit commercial use (no flag needed)

Surface the license name in the README but do not add a warning:

- `mit`
- `apache-2.0`
- `bsd-2-clause`, `bsd-3-clause`
- `cc0-1.0`
- `cc-by-2.0`, `cc-by-3.0`, `cc-by-4.0`
- `cc-by-sa-*` (share-alike obligations apply)
- `mpl-2.0`
- `unlicense`

## Unknown / missing license

When `license` is missing or set to `unknown`: surface as a warning — "License not specified on model card. Verify licensing before commercial deployment."
