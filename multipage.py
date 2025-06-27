"""Simple multipage app controller for Streamlit applications."""

import streamlit as st
from typing import Callable, List, Dict, Any


class MultiPage:
    """Simple multipage app controller for Streamlit applications."""

    def __init__(self) -> None:
        """Initialize the multipage controller."""
        self.pages: List[Dict[str, Any]] = []

    def add_page(self, title: str, func: Callable[[], None]) -> None:
        """Add a page to the multipage application.
        
        Args:
            title: Title of the page to display in the sidebar
            func: Function to call when the page is selected
        """
        self.pages.append({"title": title, "function": func})

    def run(self) -> None:
        """Execute the multipage application.
        
        Displays a sidebar with page selection and executes the selected page function.
        """
        titles = [p["title"] for p in self.pages]
        choice = st.sidebar.selectbox("Page", titles)
        for page in self.pages:
            if page["title"] == choice:
                page["function"]()
                break

