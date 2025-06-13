import streamlit as st

class MultiPage:
    """Simple multipage app controller."""

    def __init__(self) -> None:
        self.pages = []

    def add_page(self, title: str, func) -> None:
        """Add a page."""
        self.pages.append({"title": title, "function": func})

    def run(self) -> None:
        """Run the multipage app."""
        titles = [p["title"] for p in self.pages]
        choice = st.sidebar.selectbox("Page", titles)
        for page in self.pages:
            if page["title"] == choice:
                page["function"]()
                break

