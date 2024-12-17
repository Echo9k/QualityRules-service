import streamlit as st
import requests

# Streamlit setup
st.title("Rule Search Application")
st.write("Enter a description to find the most appropriate rules.")

# User input
query = st.text_area("Enter your query here:", "")
top_k = st.slider("Number of results:", min_value=1, max_value=10, value=5)

# Search button
if st.button("Search"):
    if query:
        # Call the FastAPI backend
        response = requests.post(
            "http://127.0.0.1:8000/search/",
            json={"query": query, "top_k": top_k}
        )
        if response.status_code == 200:
            results = response.json()["results"]
            st.write("### Search Results:")
            for result in results:
                st.write(f"**RuleID**: {result['RuleID']}")
                st.write(f"**Description**: {result['Description']}")
                st.write(f"**Distance**: {result['Distance']:.4f}")
                st.write("---")
        else:
            st.error("Error fetching results. Check the backend service.")
    else:
        st.warning("Please enter a query.")
