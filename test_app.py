import streamlit as st

st.title("Test App")

# タブを作成
tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])

with tab1:
    st.write("Tab 1 content")
    
with tab2:
    st.write("Tab 2 content")
    
st.write("After tabs")