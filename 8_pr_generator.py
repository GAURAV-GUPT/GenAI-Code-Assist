import streamlit as st
from git import Repo, GitCommandError
import os

# Streamlit setup
st.set_page_config(page_title="🚀 Git Commit & Push")
st.title("🚀 GenAI Code Commit & Push Assistant")

# Input fields
repo_path = st.text_input("📁 Path to your local Git repo", value="./")
file_path = st.text_input("📝 File to commit (relative to repo)", value="generated_code.py")
commit_message = st.text_input("💬 Commit message", value="feat: apply GenAI update")

# Commit and push button
if st.button("🔼 Push to Git"):
    try:
        # Check if path is a valid git repo
        repo = Repo(repo_path)
        if repo.bare:
            st.error("❌ This is not a valid Git repository.")
        else:
            # Stage the file
            full_file_path = os.path.join(repo_path, file_path)
            if not os.path.exists(full_file_path):
                st.error(f"❌ File not found: {full_file_path}")
            else:
                repo.git.add(file_path)
                repo.index.commit(commit_message)
                origin = repo.remotes.origin
                origin.push()
                st.success("✅ File committed and pushed to remote repository!")
    except GitCommandError as e:
        st.error(f"❌ Git error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")