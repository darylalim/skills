# Streamlit Docs Index

Canonical `docs.streamlit.io` URLs used by the `streamlit-app-builder` skill. Fetch these at skill-run time to verify current APIs before generating code.

## Baseline

| Topic              | URL                                                                     |
|--------------------|-------------------------------------------------------------------------|
| Multipage apps     | https://docs.streamlit.io/develop/concepts/multipage-apps/overview      |
| `st.navigation`    | https://docs.streamlit.io/develop/api-reference/navigation/st.navigation|
| `st.Page`          | https://docs.streamlit.io/develop/api-reference/navigation/st.page      |
| Caching            | https://docs.streamlit.io/develop/concepts/architecture/caching         |
| `@st.cache_resource` | https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource |
| `@st.cache_data`   | https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data |
| App testing        | https://docs.streamlit.io/develop/concepts/app-testing                  |
| `AppTest`          | https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest |
| Secrets            | https://docs.streamlit.io/develop/concepts/connections/secrets-management |
| Config             | https://docs.streamlit.io/develop/api-reference/configuration/config.toml |

## Pattern-conditional

| Widget              | URL                                                                     |
|---------------------|-------------------------------------------------------------------------|
| `st.chat_input`     | https://docs.streamlit.io/develop/api-reference/chat/st.chat_input      |
| `st.chat_message`   | https://docs.streamlit.io/develop/api-reference/chat/st.chat_message    |
| `st.audio_input`    | https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input  |
| `st.file_uploader`  | https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader|
| `st.text_input`     | https://docs.streamlit.io/develop/api-reference/widgets/st.text_input   |
| `st.text_area`      | https://docs.streamlit.io/develop/api-reference/widgets/st.text_area    |
| `st.selectbox`      | https://docs.streamlit.io/develop/api-reference/widgets/st.selectbox    |
| `st.number_input`   | https://docs.streamlit.io/develop/api-reference/widgets/st.number_input |
| `st.slider`         | https://docs.streamlit.io/develop/api-reference/widgets/st.slider       |
| `st.image`          | https://docs.streamlit.io/develop/api-reference/media/st.image          |
| `st.audio`          | https://docs.streamlit.io/develop/api-reference/media/st.audio          |
| `st.dataframe`      | https://docs.streamlit.io/develop/api-reference/data/st.dataframe       |
| `st.plotly_chart`   | https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart  |

## Root

| Area                | URL                                   |
|---------------------|---------------------------------------|
| Docs root           | https://docs.streamlit.io/            |
| API reference index | https://docs.streamlit.io/develop/api-reference |

## Out of scope

- Deployment docs (Community Cloud, Snowflake) — the skill's non-goals explicitly defer deployment to the consuming team.
- `st.connection` / database integrations — the skill does not scaffold data layers.
- Advanced theming / custom components — teams override `.streamlit/config.toml` directly per their branding.
