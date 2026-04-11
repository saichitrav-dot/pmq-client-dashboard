import streamlit as st

from modules import task8_client_pmq_dashboard


def main() -> None:
    st.set_page_config(page_title="Executive Performance Dashboard", layout="wide")
    task8_client_pmq_dashboard.run()


if __name__ == "__main__":
    main()
