import streamlit as st


def app() -> None:
    st.title("Strategy Manager")

    manager = st.session_state.strategy_manager
    st.write("Registered strategies:", list(manager.strategies.keys()))

    st.subheader("Load Strategy")
    module = st.text_input("Module Path")
    cls = st.text_input("Class Name")
    name = st.text_input("Register As")
    if st.button("Load") and module and cls and name:
        try:
            manager.load_strategy(module, cls, name)
            st.success(f"Loaded {name}")
        except Exception as e:
            st.error(str(e))

    if manager.strategies:
        st.subheader("Activate/Deactivate")
        selected = st.selectbox("Strategy", list(manager.strategies.keys()))
        col1, col2 = st.columns(2)
        if col1.button("Activate"):
            try:
                manager.activate_strategy(selected)
            except Exception as e:
                st.error(str(e))
        if col2.button("Deactivate"):
            try:
                manager.deactivate_strategy(selected)
            except Exception as e:
                st.error(str(e))

