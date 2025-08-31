import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from groq import Groq
import os
from dotenv import load_dotenv

def load_emotion_data(filepath):
    """Load and process emotion data from CSV with error handling."""
    columns = ['timestamp', 'role', 'message', 
               'emotion1', 'score1', 
               'emotion2', 'score2', 
               'emotion3', 'score3']
    
    df = pd.read_csv(
        filepath, 
        names=columns, 
        on_bad_lines='skip'
    )
    
    score_columns = ['score1', 'score2', 'score3']
    df[score_columns] = df[score_columns].apply(pd.to_numeric, errors='coerce')
    
    return df

def visualize_emotions(df):
    """Create visualizations of top 5 emotions."""
    emotion_data = []
    for column in ['emotion1', 'emotion2', 'emotion3']:
        score_col = column.replace('emotion', 'score')
        temp_df = df[[column, score_col]].rename(columns={column: 'emotion', score_col: 'score'})
        temp_df['type'] = column
        emotion_data.append(temp_df)
    
    combined_emotions = pd.concat(emotion_data)
    
    # Top 5 Emotion Frequency
    st.subheader("Top 5 Emotion Frequencies")
    emotion_counts = combined_emotions['emotion'].value_counts().head(5)
    fig_freq = px.bar(
        x=emotion_counts.index, 
        y=emotion_counts.values, 
        title="Top 5 Emotions Frequency"
    )
    st.plotly_chart(fig_freq)

    # Top 5 Emotion Intensity
    st.subheader("Top 5 Emotion Intensities")
    emotion_intensity = combined_emotions.groupby('emotion')['score'].mean().sort_values(ascending=False).head(5)
    fig_intensity = px.bar(
        x=emotion_intensity.index, 
        y=emotion_intensity.values, 
        title="Top 5 Emotions Average Intensity"
    )
    st.plotly_chart(fig_intensity)

# Rest of the code remains the same as in the previous version
def get_mental_health_analysis(df):
    """Generate mental health analysis using Groq API."""
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        st.error("Groq API key not found. Please set GROQ_API_KEY in .env file.")
        return ""
    
    client = Groq(api_key=groq_api_key)
    
    emotion_summary = f"""
    Conversation Emotion Analysis:
    - Total Messages: {len(df)}
    - Most Frequent Emotions: {df['emotion1'].value_counts().head(3).to_dict()}
    - Emotion Intensity Summary:
      1st Emotion: Average Intensity = {df['score1'].mean():.2f}
      2nd Emotion: Average Intensity = {df['score2'].mean():.2f}
      3rd Emotion: Average Intensity = {df['score3'].mean():.2f}
    
    Context: {df['message'].tolist()}
    """
    
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are a compassionate mental health assistant analyzing emotional patterns."
            },
            {
                "role": "user",
                "content": f"Provide a comprehensive mental health assessment based on these emotional patterns:\n{emotion_summary}"
            }
        ],
        max_tokens=1000,
        temperature=0.7
    )
    
    return completion.choices[0].message.content

def main():
    st.title("Emotional Wellness Analysis")
    
    file_path = 'emotion_analysis.csv'
    
    try:
        df = load_emotion_data(file_path)
        
        st.header("Emotion Visualizations")
        visualize_emotions(df)
        
        st.header("Mental Health Insights")
        if st.button("Generate Mental Health Report"):
            with st.spinner('Analyzing emotional patterns...'):
                report = get_mental_health_analysis(df)
                st.write(report)
    
    except Exception as e:
        st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()