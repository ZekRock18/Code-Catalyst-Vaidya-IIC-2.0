import streamlit as st
import asyncio
import os
import base64
import threading
import queue
import csv
from datetime import datetime, timezone
from dotenv import load_dotenv

from hume.client import AsyncHumeClient
from hume.empathic_voice.chat.socket_client import ChatConnectOptions, ChatWebsocketConnection
from hume.empathic_voice.chat.types import SubscribeEvent
from hume.core.api_error import ApiError
from hume import MicrophoneInterface, Stream

class StreamlitWebSocketHandler:
    def __init__(self, csv_filename='emotion_analysis.csv'):
        self.messages_queue = queue.Queue()
        self.emotions_queue = queue.Queue()
        self.socket = None
        self.byte_strs = Stream.new()
        self.stop_event = threading.Event()
        self.csv_filename = csv_filename

    def set_socket(self, socket: ChatWebsocketConnection):
        """Set the socket."""
        self.socket = socket

    def _extract_top_n_emotions(self, emotion_scores: dict, n: int) -> dict:
        """Extract the top N emotions based on confidence scores."""
        sorted_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)
        return {emotion: score for emotion, score in sorted_emotions[:n]}

    def _log_to_csv(self, role: str, message: str, emotions: dict):
        """Log emotions to CSV file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare emotion data (pad with empty strings if fewer than 3 emotions)
        emotion_data = []
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        for i in range(3):
            if i < len(sorted_emotions):
                emotion_data.extend([sorted_emotions[i][0], sorted_emotions[i][1]])
            else:
                emotion_data.extend(['', ''])
        
        # Write to CSV
        with open(self.csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csvfile.write(f"{timestamp},{role},{message},{emotion_data[0]},{emotion_data[1]},{emotion_data[2]},{emotion_data[3]}\n")

    async def on_message(self, message: SubscribeEvent):
        """Callback function to handle WebSocket message events."""
        scores = {}

        if message.type in ["user_message", "assistant_message"]:
            role = message.message.role.upper()
            message_text = message.message.content
            
            if message.from_text is False:
                scores = dict(message.models.prosody.scores)
            
            # Add message to queue
            self.messages_queue.put(f"{role}: {message_text}")
            
            # Add top 3 emotions to queue if available
            if len(scores) > 0:
                top_3_emotions = self._extract_top_n_emotions(scores, 3)
                self.emotions_queue.put(top_3_emotions)
                
                # Log to CSV
                self._log_to_csv(role, message_text, top_3_emotions)
        
        elif message.type == "audio_output":
            message_str: str = message.data
            message_bytes = base64.b64decode(message_str.encode("utf-8"))
            await self.byte_strs.put(message_bytes)

    async def on_open(self):
        """Logic invoked when the WebSocket connection is opened."""
        st.toast("WebSocket connection opened.")

    async def on_close(self):
        """Logic invoked when the WebSocket connection is closed."""
        st.toast("WebSocket connection closed.")
        self.stop_event.set()

    async def on_error(self, error):
        """Logic invoked when an error occurs in the WebSocket connection."""
        st.error(f"Error: {error}")
        self.stop_event.set()

def emotion_analysis_app():
    st.title("Emotion Analysis Chat")

    # Environment setup
    load_dotenv()
    HUME_API_KEY = os.getenv("HUME_API_KEY")
    HUME_SECRET_KEY = os.getenv("HUME_SECRET_KEY")
    HUME_CONFIG_ID = os.getenv("HUME_CONFIG_ID")

    # Check API keys
    if not all([HUME_API_KEY, HUME_SECRET_KEY, HUME_CONFIG_ID]):
        st.error("Please set HUME_API_KEY, HUME_SECRET_KEY, and HUME_CONFIG_ID in .env file")
        return

    # Initialize session state
    if 'websocket_handler' not in st.session_state:
        st.session_state.websocket_handler = StreamlitWebSocketHandler()
    
    # Messages and emotions display
    messages_placeholder = st.container()
    emotions_placeholder = st.container()

    # Store messages and emotions
    if 'stored_messages' not in st.session_state:
        st.session_state.stored_messages = []
    if 'stored_emotions' not in st.session_state:
        st.session_state.stored_emotions = []

    async def start_emotion_analysis():
        client = AsyncHumeClient(api_key=HUME_API_KEY)
        options = ChatConnectOptions(config_id=HUME_CONFIG_ID, secret_key=HUME_SECRET_KEY)
        
        async with client.empathic_voice.chat.connect_with_callbacks(
            options=options,
            on_open=st.session_state.websocket_handler.on_open,
            on_message=st.session_state.websocket_handler.on_message,
            on_close=st.session_state.websocket_handler.on_close,
            on_error=st.session_state.websocket_handler.on_error
        ) as socket:
            st.session_state.websocket_handler.set_socket(socket)

            microphone_task = asyncio.create_task(
                MicrophoneInterface.start(
                    socket,
                    allow_user_interrupt=False,
                    byte_stream=st.session_state.websocket_handler.byte_strs
                )
            )
            
            # Process messages and emotions
            while not st.session_state.websocket_handler.stop_event.is_set():
                # Process messages
                while not st.session_state.websocket_handler.messages_queue.empty():
                    message = st.session_state.websocket_handler.messages_queue.get()
                    st.session_state.stored_messages.append(message)
                
                # Process emotions
                while not st.session_state.websocket_handler.emotions_queue.empty():
                    emotion = st.session_state.websocket_handler.emotions_queue.get()
                    st.session_state.stored_emotions.append(emotion)
                
                await asyncio.sleep(0.1)

    # Render messages and emotions
    messages_placeholder.empty()
    emotions_placeholder.empty()

    with messages_placeholder:
        st.subheader("Messages")
        for msg in st.session_state.stored_messages:
            st.write(msg)

    with emotions_placeholder:
        st.subheader("Emotion Analysis")
        for emotion_dict in st.session_state.stored_emotions:
            emotion_text = " | ".join([f"{emotion} ({score:.2f})" for emotion, score in emotion_dict.items()])
            st.write(emotion_text)

    # Run button
    if st.button("Start Emotion Analysis"):
        # Reset stored messages and emotions
        st.session_state.stored_messages = []
        st.session_state.stored_emotions = []
        
        # Run the async function
        asyncio.run(start_emotion_analysis())

def main():
    emotion_analysis_app()

if __name__ == "__main__":
    main()