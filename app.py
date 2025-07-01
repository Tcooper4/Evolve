"""
DEPRECATED: Main Streamlit Application

This entry point is deprecated. Please use unified_interface.py instead.
This file now redirects to the enhanced unified interface.
"""

import streamlit as st
import sys
from pathlib import Path

# Add deprecation warning
st.warning("""
⚠️ **DEPRECATED ENTRY POINT**

This entry point (app.py) is deprecated and will be removed in a future version.
Please use `unified_interface.py` instead for the enhanced trading system.

**To run the system:**
```bash
streamlit run unified_interface.py
```
""")

# Redirect to unified interface
st.info("Redirecting to unified interface...")

# Import and run the unified interface
try:
    from unified_interface import run_enhanced_interface
    result = run_enhanced_interface()
    st.success("Successfully redirected to unified interface")
except ImportError as e:
    st.error(f"Failed to import unified interface: {e}")
    st.info("Please ensure unified_interface.py is available")
except Exception as e:
    st.error(f"Error running unified interface: {e}")
    st.info("Please check the unified interface for errors")
