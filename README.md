# Data Validation Tool

This repository contains a Streamlit application for validating time-series economic indicators. The tool streamlines the process of cleaning input files, mapping indicator names, and running automated quality checks. It is designed for analysts who routinely receive spreadsheets with inconsistent headers, varied date formats, and potential data anomalies.

## Features
- **Header detection and renaming**: Detects the appropriate header row and maps column names to canonical indicator labels.
- **Synonym management**: Matches incoming column headers against a curated list of indicator synonyms and typos.
- **Wide-to-long reshaping**: Converts spreadsheets with year-based wide layouts into tidy long-form data automatically.
- **Data coercion**: Attempts to coerce numeric indicator columns and identifies non-numeric entries.
- **Diagnostics**: Runs gap detection, outlier checks, and structural-break scans across grouped series.
- **Custom rules**: Supports authoring bespoke numeric relationships to validate business logic.

## Getting Started

### Requirements
- Python 3.9 or newer
- [Streamlit](https://streamlit.io)
- pandas 2.2 or newer
- numpy
- matplotlib
- statsmodels
- st-aggrid

All Python dependencies used by the app are listed in [`requirements.txt`](requirements.txt).

### Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run DVT_github_V2.py
```

When launched, the Streamlit app guides you through uploading a dataset, reviewing detected column mappings, applying transformations, and downloading validated outputs.

### Repository Structure
- `DVT_github_V2.py` – main Streamlit application.
- `requirements.txt` – Python dependencies for local development and deployment.

## Development Tips
- The app uses `st.cache_data` for expensive operations such as file sniffing. Clear the cache from the Streamlit sidebar if you need to force a reload.
- When updating indicator synonyms, ensure both the canonical map and lower-case lookup dictionary remain in sync.
- Use the built-in plots and logs to review anomalies detected by the validation routines.

## License
This project currently does not specify a license. Please add one if you plan to distribute the tool externally.
