# app.py
import streamlit as st
import random
import json
import time

# --- Simulated Sensor Data ---
def get_sensor_data():
    """Generates a dictionary of simulated car sensor data."""
    return {
        "engine_temperature": f"{random.randint(85, 100)}°C",
        "tire_pressure_fl": f"{random.randint(30, 35)} PSI",
        "tire_pressure_fr": f"{random.randint(30, 35)} PSI",
        "tire_pressure_rl": f"{random.randint(30, 35)} PSI",
        "tire_pressure_rr": f"{random.randint(30, 35)} PSI",
        "oil_level": f"{random.randint(70, 100)}%",
        "brake_fluid_level": f"{random.randint(80, 100)}%",
        "battery_voltage": f"{random.randint(11, 14)}V",
        "fuel_level": f"{random.randint(10, 95)}L",
    }

# --- Streamlit App Layout and Logic ---

# Initialize session state for chat history and sensor data
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sensor_data" not in st.session_state:
    st.session_state.sensor_data = get_sensor_data()

st.set_page_config(
    page_title="Gen AI Car Diagnostic",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.markdown("<h1 style='text-align: center;'>Gen AI Car Diagnostic</h1>", unsafe_allow_html=True)

# Create two columns for the layout
col1, col2 = st.columns([1, 2], gap="large")

# Left Column: Sensor Data Dashboard
with col1:
    st.markdown("<h2 style='text-align: center; color: #4b5563;'>Vehicle Dashboard</h2>", unsafe_allow_html=True)
    
    # Function to get Font Awesome icons
    def get_icon_for_key(key):
        icons = {
            'engine_temperature': 'thermometer-half',
            'tire_pressure_fl': 'car-tire-four',
            'tire_pressure_fr': 'car-tire-four',
            'tire_pressure_rl': 'car-tire-four',
            'tire_pressure_rr': 'car-tire-four',
            'oil_level': 'oil-can',
            'brake_fluid_level': 'fill-drip',
            'battery_voltage': 'car-battery',
            'fuel_level': 'gas-pump',
        }
        return icons.get(key, 'info-circle')

    # Display sensor data using markdown and columns for better layout
    for key, value in st.session_state.sensor_data.items():
        st.markdown(f"""
        <div style="background-color: #f3f4f6; padding: 16px; border-radius: 12px; margin-bottom: 8px; display: flex; align-items: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <i class="fa-solid fa-{get_icon_for_key(key)}" style="font-size: 24px; color: #3b82f6; margin-right: 16px;"></i>
            <div>
                <p style="font-size: 14px; color: #6b7280; margin: 0; text-transform: capitalize;">{key.replace('_', ' ')}</p>
                <p style="font-size: 18px; font-weight: 600; color: #1f2937; margin: 0;">{value}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        

# Right Column: Chat Interface
with col2:
    st.markdown("<h2 style='text-align: center; color: #4b5563;'>AI Diagnostic Partner</h2>", unsafe_allow_html=True)
    
    # Display initial welcome message
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Welcome! I'm your AI Diagnostic Partner. Ask me about your car's health."})

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    prompt = st.chat_input("Type your question...")
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get the AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                time.sleep(random.uniform(1, 3)) # Simulate AI latency

                # --- Gen AI Logic (Simulated) ---
                # This is the same simulated logic from the Flask app
                simulated_response = "I'm analyzing your car's data right now. Everything looks good at the moment. What's on your mind?"
                
                sensor_data = st.session_state.sensor_data

                if "tire pressure" in prompt.lower() or "tires" in prompt.lower():
                    if any(int(v.split(' ')[0]) < 32 for k, v in sensor_data.items() if 'tire_pressure' in k):
                        simulated_response = "I've detected that one or more of your tires has low pressure. I recommend checking and inflating them to the manufacturer's recommended PSI soon to ensure safe driving and fuel efficiency."
                    else:
                        simulated_response = "Your tire pressure is currently within the optimal range. They're all set!"

                elif "oil" in prompt.lower():
                    if int(sensor_data['oil_level'].split('%')[0]) < 80:
                        simulated_response = "Your oil level is a bit low. It's a good idea to top it off when you have a chance to prevent any potential engine issues."
                    else:
                        simulated_response = "Your oil level is healthy and doesn't require any immediate attention."
                        
                elif "engine" in prompt.lower() or "temperature" in prompt.lower():
                    if int(sensor_data['engine_temperature'].split('°')[0]) > 95:
                        simulated_response = "I'm seeing a slightly elevated engine temperature. It might be worth checking your coolant levels. If the warning light comes on, please pull over safely."
                    else:
                        simulated_response = "Your engine temperature is currently stable and in the normal operating range. No issues there!"

                elif "problem" in prompt.lower() or "issue" in prompt.lower():
                    simulated_response = "Based on the current data, I'm not seeing any red flags. Can you describe the issue you're experiencing, and I'll see if I can cross-reference it with the sensor data?"

                st.write(simulated_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": simulated_response})
