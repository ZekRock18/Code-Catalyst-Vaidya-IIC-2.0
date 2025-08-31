import streamlit as st
from models import import_app_from_file
from pathlib import Path

def app():
    st.title("Mental Health and Emotion Analysis Tools")
    
    # Dictionary mapping display names to script paths
    emotion_apps = {
        "Emotion Analysis Chat": "mental_health.py",
        "Mental Health Insights": "mhanalysis.py"
    }
    
    # Create the dropdown for app selection
    selected_app = st.selectbox(
        "Select Application",
        options=list(emotion_apps.keys()),
        index=None,
        placeholder="Choose an application..."
    )
    
    # Load and run the selected app
    if selected_app:
        script_path = emotion_apps[selected_app]
        
        if Path(script_path).exists():
            # Import the selected app module
            app_module = import_app_from_file(script_path)
            
            if app_module and hasattr(app_module, 'main'):
                # Create a visual separator
                st.markdown("---")
                
                # Run the app's main function
                app_module.main()
            else:
                st.error(f"The selected app doesn't have the required 'main' function")
        else:
            st.error(f"App file not found: {script_path}")

