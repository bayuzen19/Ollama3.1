import ollama
import streamlit as st

# Set page configuration
st.set_page_config(page_title="ZenBot", page_icon="🤖", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        font-family: Arial, sans-serif;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        transition-duration: 0.4s;
    }
    .stButton button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    .stTextArea textarea {
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    .message {
        background-color: #e9e9e9;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #d1e7dd;
        text-align: right;
    }
    .bot-message {
        background-color: #f0f4ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title with an icon
st.title("ZenBot 🤖")

# Initialize session state to track conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# User prompt input
prompt = st.text_area(label="Your message:", placeholder="Enter your message here...")

# Submit button
if st.button("Send"):
    if prompt:
        # Append the current prompt to the conversation history
        st.session_state.conversation.append({"sender": "user", "message": prompt})
        
        # Construct the conversation history as a single string to send to the model
        conversation_history = "\n".join([f"{chat['sender']}: {chat['message']}" for chat in st.session_state.conversation])
        
        # Generate response from the bot, including the entire conversation history
        response = ollama.generate(model="llama3.1", prompt=conversation_history)
        
        # Append the bot's response to the conversation history
        st.session_state.conversation.append({"sender": "bot", "message": response["response"]})
        
        # Clear the input box after submission
        st.text_area(label="Your message:", placeholder="Enter your message here...", value="", key="reset")
    else:
        st.warning("Please enter a message before sending!")

# Display conversation history
for chat in st.session_state.conversation:
    sender = chat.get("sender", "")
    message = chat.get("message", "")
    
    if sender == "user":
        st.markdown(f'<div class="message user-message">{message}</div>', unsafe_allow_html=True)
    elif sender == "bot":
        st.markdown(f'<div class="message bot-message">{message}</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <hr style="border: 1px solid #ddd;">
    <div style="text-align: center; font-size: 12px; color: #aaa;">
        © 2024 ZenBot. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True,
)
