import sys
from pathlib import Path

import streamlit as st


APP_ROOT = Path(__file__).resolve().parent / "test"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from modules import task8_client_pmq_dashboard


def main() -> None:
    st.set_page_config(page_title="Executive Performance Dashboard", layout="wide")
    task8_client_pmq_dashboard.run()


if __name__ == "__main__":
    main()
