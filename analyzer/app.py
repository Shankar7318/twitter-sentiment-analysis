import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import nltk

# Set page config first
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)

# Try to import required packages
try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False

try:
    from keras.preprocessing.sequence import pad_sequences
    PAD_SEQUENCES_AVAILABLE = True
except ImportError:
    PAD_SEQUENCES_AVAILABLE = False
    st.sidebar.warning("pad_sequences not available")

try:
    from keras.models import load_model
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    st.sidebar.warning("Keras not available")

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    with st.spinner("Downloading NLTK data..."):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Initialize lemmatizer
lem = WordNetLemmatizer()

# Text preprocessing function (same as your training)
def preprocess_text(text):
    try:
        if not isinstance(text, str):
            return ""
            
        # Remove @tags
        text = re.sub(r"@\S+", ' ', text)
        
        # Smart lowercase
        text = text.lower()
        
        # Remove numbers
        text = re.sub(r'\d+', ' ', text)
        
        # Remove links
        text = re.sub(r"https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', text)
        
        # Remove Punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove white spaces
        text = text.strip()
        
        # Tokenize into words
        tokens = word_tokenize(text)
        
        # Remove non alphabetic tokens
        tokens = [word for word in tokens if word.isalpha()]
        
        # Filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        
        # Word Lemmatization
        tokens = [lem.lemmatize(word, "v") for word in tokens]
        
        # Turn lists back to string
        processed_text = ' '.join(tokens)
        
        return processed_text
        
    except Exception as e:
        st.error(f"Error in text preprocessing: {e}")
        return text

# Load models with error handling
@st.cache_resource
def load_models():
    tokenizer = None
    model = None
    models_loaded = False
    
    # Load tokenizer
    try:
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        st.sidebar.success("✅ Tokenizer loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Tokenizer error: {e}")
    
    # Load model
    try:
        if KERAS_AVAILABLE:
            model = load_model('minimal_model.h5')
            st.sidebar.success("✅ Model loaded")
            models_loaded = True
        else:
            st.sidebar.warning("🤖 Keras not available - using rule-based analysis")
    except Exception as e:
        st.sidebar.error(f"❌ Model error: {e}")
    
    return tokenizer, model, models_loaded

# Simple fallback sentiment analysis
def simple_sentiment_analysis(text):
    positive_words = ['good', 'great', 'excellent', 'amazing', 'happy', 'love', 'nice', 'best', 'awesome', 'fantastic', 'wonderful', 'perfect', 'outstanding']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'sad', 'angry', 'disappointing', 'hate', 'dislike', 'poor', 'annoying']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    total_words = positive_count + negative_count
    if total_words > 0:
        score = positive_count / total_words
    else:
        score = 0.5
    
    # Adjust score based on overall sentiment strength
    if positive_count > negative_count:
        score = 0.5 + (score * 0.5)  # Map to 0.5-1.0
    elif negative_count > positive_count:
        score = score * 0.5  # Map to 0.0-0.5
    
    if score >= 0.6:
        label = "Positive"
        emoji_icon = "😊"
        color = "green"
    elif score <= 0.4:
        label = "Negative"
        emoji_icon = "😞"
        color = "red"
    else:
        label = "Neutral"
        emoji_icon = "😐"
        color = "orange"
    
    return {
        "label": label,
        "score": score,
        "emoji": emoji_icon,
        "color": color,
        "processed_text": text,
        "method": "Rule-based"
    }

# Advanced prediction with model
def predict_with_model(text, tokenizer, model, max_sequence_length=300):
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        if not PAD_SEQUENCES_AVAILABLE:
            st.warning("pad_sequences not available - using rule-based")
            return simple_sentiment_analysis(text)
        
        # Convert to sequence and pad
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
        
        # Make prediction
        score = model.predict(padded_sequence, verbose=0)[0][0]
        
        # Determine sentiment
        if score >= 0.6:
            label = "Positive"
            emoji_icon = "😊"
            color = "green"
        elif score <= 0.4:
            label = "Negative" 
            emoji_icon = "😞"
            color = "red"
        else:
            label = "Neutral"
            emoji_icon = "😐"
            color = "orange"
        
        return {
            "label": label,
            "score": float(score),
            "emoji": emoji_icon,
            "color": color,
            "processed_text": processed_text,
            "method": "AI Model"
        }
        
    except Exception as e:
        st.error(f"Model prediction error: {e}")
        return simple_sentiment_analysis(text)

# Function to detect text column in CSV
def detect_text_column(df):
    """Automatically detect which column contains text data"""
    text_columns = []
    
    # Common column names that might contain text
    common_text_names = [
        'text', 'tweet', 'review', 'comment', 'message', 'content', 
        'description', 'feedback', 'post', 'sentence', 'phrase',
        'tweets', 'reviews', 'comments', 'messages', 'contents',
        'descriptions', 'feedback', 'posts', 'sentences'
    ]
    
    # Check for exact matches first
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in common_text_names:
            text_columns.append((col, 10))  # High priority for exact matches
    
    # Check for partial matches
    for col in df.columns:
        col_lower = col.lower().strip()
        for common_name in common_text_names:
            if common_name in col_lower and col not in [tc[0] for tc in text_columns]:
                text_columns.append((col, 5))  # Medium priority for partial matches
    
    # Check data type and content for remaining columns
    for col in df.columns:
        if col not in [tc[0] for tc in text_columns]:
            # Check if column contains string data and has reasonable text content
            if df[col].dtype == 'object':
                sample_value = str(df[col].iloc[0]) if len(df) > 0 else ""
                if len(sample_value) > 10 and len(sample_value) < 500:  # Reasonable text length
                    text_columns.append((col, 1))  # Low priority based on content
    
    # Sort by priority and return the best column
    if text_columns:
        text_columns.sort(key=lambda x: x[1], reverse=True)
        return text_columns[0][0]
    
    return None

def main():
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Load models
    tokenizer, model, models_loaded = load_models()
    
    # Analysis mode
    analysis_mode = st.sidebar.radio(
        "Analysis Mode:",
        ["Auto (Recommended)", "AI Model Only", "Rule-Based Only"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **How it works:**
    - **AI Model**: Uses LSTM neural network
    - **Rule-Based**: Word pattern analysis
    - **Auto**: Uses AI when available, falls back to rules
    """)
    
    # Main content
    st.title("😊 Text Sentiment Analysis")
    st.markdown("Analyze the emotional tone of your text using AI")
    
    # Example texts
    examples = {
        "Choose an example": "",
        "😊 Positive Review": "I love this product! It's amazing and works perfectly. Highly recommended! Will definitely buy again!",
        "😞 Negative Review": "This is terrible. Worst customer service ever. Very disappointed with the quality and support.",
        "😐 Neutral Statement": "The package arrived on Tuesday. It was as described in the product listing.",
        "Mixed Emotions": "she went outside. it was raining and she was happy. but she had to come back home so that made her sad",
        "Excited Customer": "Absolutely fantastic! The service exceeded all my expectations. Quick delivery and excellent quality!"
    }
    
    selected_example = st.selectbox("Try an example:", list(examples.keys()))
    
    if selected_example != "Choose an example":
        default_text = examples[selected_example]
    else:
        default_text = ""
    
    # Text input
    user_text = st.text_area(
        "**Enter your text for analysis:**",
        value=default_text,
        height=150,
        placeholder="Type or paste your text here...",
        help="The text will be preprocessed (cleaned, tokenized, lemmatized) before analysis"
    )
    
    # Analysis options
    col1, col2 = st.columns(2)
    with col1:
        show_processed = st.checkbox("Show processed text", help="View how the text is cleaned and prepared")
    with col2:
        show_details = st.checkbox("Show detailed analysis", value=True)
    
    # Analyze button
    if st.button("🎯 Analyze Sentiment", type="primary", use_container_width=True):
        if not user_text or not user_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
            return
            
        with st.spinner("🔍 Analyzing sentiment..."):
            try:
                # Determine analysis method
                if analysis_mode == "AI Model Only" and models_loaded and tokenizer and model:
                    result = predict_with_model(user_text, tokenizer, model)
                elif analysis_mode == "Rule-Based Only":
                    result = simple_sentiment_analysis(user_text)
                else:  # Auto mode
                    if models_loaded and tokenizer and model:
                        result = predict_with_model(user_text, tokenizer, model)
                    else:
                        result = simple_sentiment_analysis(user_text)
                        st.info("🤖 Using rule-based analysis (AI model not available)")
                
                # Display results
                st.subheader("📊 Analysis Results")
                
                # Create result columns
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric("**Sentiment**", f"{result['emoji']} {result['label']}")
                
                with res_col2:
                    st.metric("**Confidence**", f"{result['score']:.4f}")
                
                with res_col3:
                    st.metric("**Method**", result['method'])
                
                # Visual progress bar
                st.write("**Confidence Level:**")
                if result['label'] == "Positive":
                    progress_value = result['score']
                    progress_color = "green"
                elif result['label'] == "Negative":
                    progress_value = 1 - result['score']
                    progress_color = "red"
                else:
                    progress_value = 0.5
                    progress_color = "orange"
                
                st.progress(float(progress_value))
                
                # Color-coded result
                if result['color'] == 'green':
                    st.success(f"## {result['emoji']} Positive Sentiment")
                    st.balloons()
                elif result['color'] == 'red':
                    st.error(f"## {result['emoji']} Negative Sentiment")
                else:
                    st.warning(f"## {result['emoji']} Neutral Sentiment")
                
                # Show processed text if requested
                if show_processed:
                    with st.expander("🔍 View Text Processing Details"):
                        st.write("**Original Text:**")
                        st.info(user_text)
                        st.write("**Processed Text (after cleaning):**")
                        st.success(result['processed_text'])
                
                # Detailed insights
                if show_details:
                    st.subheader("💡 Detailed Insights")
                    
                    if result['label'] == "Positive":
                        st.success("""
                        **Positive Sentiment Detected!**
                        
                        ✅ **Characteristics:**
                        - Expresses satisfaction and happiness
                        - Contains favorable opinions
                        - Good for testimonials and reviews
                        
                        💡 **Use Cases:**
                        - Customer feedback analysis
                        - Review monitoring
                        - Brand sentiment tracking
                        """)
                    elif result['label'] == "Negative":
                        st.error("""
                        **Negative Sentiment Detected!**
                        
                        ⚠️ **Characteristics:**
                        - Expresses dissatisfaction or concerns
                        - May indicate problems or issues
                        - Opportunity for improvement
                        
                        💡 **Recommendations:**
                        - Address customer concerns
                        - Monitor for recurring issues
                        - Consider proactive support
                        """)
                    else:
                        st.warning("""
                        **Neutral Sentiment Detected!**
                        
                        🔄 **Characteristics:**
                        - Balanced or factual content
                        - May contain mixed emotions
                        - Often informational or descriptive
                        
                        💡 **Interpretation:**
                        - Could be objective reporting
                        - May need more context
                        - Common in news or descriptions
                        """)
                        
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("🔄 Trying fallback analysis...")
                result = simple_sentiment_analysis(user_text)
                
                # Display fallback results
                st.warning(f"**Fallback Analysis:** {result['emoji']} {result['label']}")
                st.write(f"**Score:** {result['score']:.4f}")
    
    # Batch analysis section
    st.markdown("---")
    st.subheader("📁 Batch Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV file for batch analysis", 
        type=['csv'],
        help="Upload a CSV file containing text data. The app will automatically detect the text column."
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded successfully! ({len(df)} rows, {len(df.columns)} columns)")
            
            st.write("**File Preview:**")
            st.dataframe(df.head(3))
            
            # Auto-detect text column
            text_column = detect_text_column(df)
            
            if text_column:
                st.success(f"🔍 Auto-detected text column: **'{text_column}'**")
                
                # Show column selection option
                st.write("**Column Selection:**")
                selected_column = st.selectbox(
                    "Choose the column to analyze:",
                    options=df.columns,
                    index=list(df.columns).index(text_column),
                    help="Select which column contains the text you want to analyze"
                )
                
                if st.button("🚀 Analyze All Texts", use_container_width=True):
                    with st.spinner(f"Analyzing {len(df)} texts from column '{selected_column}'..."):
                        results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, text in enumerate(df[selected_column]):
                            status_text.text(f"Processing {i+1}/{len(df)}: {str(text)[:50]}...")
                            
                            if pd.isna(text) or text == "":
                                result = {
                                    "label": "Unknown", 
                                    "score": 0.5, 
                                    "emoji": "❓",
                                    "method": "Skipped (empty)"
                                }
                            else:
                                if models_loaded and tokenizer and model:
                                    result = predict_with_model(str(text), tokenizer, model)
                                else:
                                    result = simple_sentiment_analysis(str(text))
                            
                            results.append(result)
                            progress_bar.progress((i + 1) / len(df))
                        
                        status_text.text("✅ Analysis complete!")
                        
                        # Create results dataframe
                        results_df = df.copy()
                        results_df['sentiment_label'] = [r['label'] for r in results]
                        results_df['sentiment_score'] = [r['score'] for r in results]
                        results_df['sentiment_emoji'] = [r['emoji'] for r in results]
                        results_df['analysis_method'] = [r['method'] for r in results]
                        
                        st.subheader("📈 Batch Results")
                        
                        # Show results with original data
                        st.dataframe(results_df)
                        
                        # Summary statistics
                        st.subheader("📊 Summary Statistics")
                        sentiment_counts = results_df['sentiment_label'].value_counts()
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total Texts", len(results_df))
                        with col2:
                            st.metric("Positive", sentiment_counts.get('Positive', 0))
                        with col3:
                            st.metric("Negative", sentiment_counts.get('Negative', 0))
                        with col4:
                            st.metric("Neutral", sentiment_counts.get('Neutral', 0))
                        with col5:
                            st.metric("Unknown", sentiment_counts.get('Unknown', 0))
                        
                        # Visualization
                        st.subheader("📊 Visualization")
                        
                        if len(sentiment_counts) > 0:
                            chart_data = pd.DataFrame({
                                'Sentiment': sentiment_counts.index,
                                'Count': sentiment_counts.values
                            })
                            st.bar_chart(chart_data.set_index('Sentiment'))
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Results as CSV",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Download summary
                        summary_data = {
                            'Statistic': ['Total Texts', 'Positive', 'Negative', 'Neutral', 'Unknown'],
                            'Count': [
                                len(results_df),
                                sentiment_counts.get('Positive', 0),
                                sentiment_counts.get('Negative', 0),
                                sentiment_counts.get('Neutral', 0),
                                sentiment_counts.get('Unknown', 0)
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        summary_csv = summary_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📊 Download Summary as CSV",
                            data=summary_csv,
                            file_name="sentiment_summary.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
            else:
                st.error("❌ Could not automatically detect a text column in your CSV file.")
                st.info("""
                **Tips for your CSV file:**
                - Name your text column as: 'text', 'tweet', 'review', 'comment', or 'message'
                - Ensure the column contains actual text content
                - Avoid very short or very long column names
                - Make sure the column has string/text data
                """)
                
                st.write("**Available columns in your file:**")
                for i, col in enumerate(df.columns):
                    st.write(f"{i+1}. `{col}` (dtype: {df[col].dtype})")
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

if __name__ == "__main__":
    main()