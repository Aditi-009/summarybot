import streamlit as st
import pandas as pd
import os
import io
from pathlib import Path
import json
from datetime import datetime
import sys

# Import the backend module
from backend import NewsSummarizerBot, get_available_columns

# Configure Streamlit page
st.set_page_config(
    page_title="News Summarizer Bot",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    return api_key.startswith('sk-') and len(api_key) > 20

def save_uploaded_file(uploaded_file, upload_dir: str = "uploads") -> str:
    """Save uploaded file to disk and return path"""
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def display_file_preview(df: pd.DataFrame, text_column: str):
    """Display a preview of the uploaded file"""
    st.subheader("📄 File Preview")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric("Total Rows", len(df))
        st.metric("Total Columns", len(df.columns))
    
    with col2:
        st.metric("Text Column", text_column)
        if text_column in df.columns:
            avg_text_length = df[text_column].astype(str).str.len().mean()
            st.metric("Avg Text Length", f"{avg_text_length:.0f} chars")
    
    # Display first few rows
    st.write("**Sample Data:**")
    preview_df = df.head(3)
    if text_column in preview_df.columns:
        # Truncate long text for display
        preview_df[text_column] = preview_df[text_column].astype(str).str[:200] + "..."
    
    st.dataframe(preview_df, use_container_width=True)

def display_processing_progress():
    """Display processing progress"""
    progress_container = st.container()
    
    with progress_container:
        st.info("🔄 Processing your news file...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate progress (in real implementation, you'd get actual progress from backend)
        import time
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 20:
                status_text.text("Analyzing file structure and extracting metadata...")
            elif i < 30:
                status_text.text("Identifying company information and sources...")
            elif i < 60:
                status_text.text("Summarizing individual news items...")
            elif i < 90:
                status_text.text("Creating overall summary with metadata...")
            else:
                status_text.text("Finalizing results...")
            time.sleep(0.05)
        
        status_text.text("✅ Processing complete!")

def display_company_info(results: dict):
    """Display company information and metadata"""
    company_info = results.get("company_info", {})
    date_range = results.get("date_range", {})
    
    if any([company_info.get('company_name'), company_info.get('ticker'), date_range.get('start_date')]):
        st.subheader("🏢 Company Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if company_info.get('company_name'):
                st.info(f"**Company:** {company_info['company_name']}")
            else:
                st.info("**Company:** Not identified")
        
        with col2:
            if company_info.get('ticker'):
                st.info(f"**Ticker:** {company_info['ticker']}")
            else:
                st.info("**Ticker:** Not identified")
        
        with col3:
            if date_range.get('start_date'):
                if date_range['start_date'] == date_range.get('end_date', ''):
                    st.info(f"**Date:** {date_range['start_date']}")
                else:
                    st.info(f"**Date Range:** {date_range['start_date']} to {date_range.get('end_date', '')}")
            else:
                st.info("**Date:** Not identified")
        
        st.divider()

def display_sources_info(results: dict):
    """Display top news sources"""
    top_sources = results.get("top_sources", [])
    
    if top_sources:
        st.subheader("📊 Top News Sources")
        
        # Create a more visual display of sources
        for i, (source, count) in enumerate(top_sources, 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{i}. {source}**")
            with col2:
                st.metric("Articles", count)
        
        st.divider()

def display_results(results: dict):
    """Display processing results"""
    if results.get("error"):
        st.error(f"❌ Error: {results['error']}")
        return
    
    st.success("✅ Processing completed successfully!")
    
    # Display document title prominently
    document_title = results.get("document_title", "News Summary Report")
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #1f77b4, #2e8b57);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        ">
            <h2 style="margin: 0; color: white;">{document_title}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display company information
    display_company_info(results)
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Items Processed", results.get("processed_items", 0))
    
    with col2:
        st.metric("Text Column Used", results.get("text_column_used", "N/A"))
    
    with col3:
        source_col = results.get("source_column_used", "Not found")
        st.metric("Source Column", source_col if source_col else "Not found")
    
    with col4:
        if "output_file" in results:
            file_size = os.path.getsize(results["output_file"]) / 1024  # KB
            st.metric("Output File Size", f"{file_size:.1f} KB")
    
    st.divider()
    
    # Display top news sources
    display_sources_info(results)
    
    # Display overall summary
    st.subheader("📋 Overall News Summary")
    overall_summary = results.get("overall_summary", "No summary available")
    
    # Create a styled container for the summary
    st.markdown(
        f"""
        <div style="
            background-color: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="line-height: 1.6; color: #333;">
                {overall_summary.replace('\n', '<br>')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Display individual summaries
    with st.expander("📑 View Individual Summaries", expanded=False):
        individual_summaries = results.get("individual_summaries", [])
        
        if individual_summaries:
            # Add search/filter functionality
            search_term = st.text_input("🔍 Search summaries:", placeholder="Enter keywords to filter summaries...")
            
            filtered_summaries = individual_summaries
            if search_term:
                filtered_summaries = [s for s in individual_summaries if search_term.lower() in s.lower()]
                st.info(f"Found {len(filtered_summaries)} summaries matching '{search_term}'")
            
            for i, summary in enumerate(filtered_summaries, 1):
                with st.container():
                    st.markdown(f"**📄 Item {i}:**")
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #fafafa;
                            padding: 15px;
                            border-radius: 8px;
                            border-left: 3px solid #28a745;
                            margin: 10px 0;
                        ">
                            {summary}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.write("No individual summaries available.")
    
    # Download section
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "output_file" in results and os.path.exists(results["output_file"]):
            with open(results["output_file"], "rb") as file:
                filename = os.path.basename(results["output_file"])
                st.download_button(
                    label="📊 Download Detailed CSV",
                    data=file.read(),
                    file_name=filename,
                    mime="text/csv",
                    help="Contains original data with individual summaries",
                    use_container_width=True
                )
    
    with col2:
        if "overall_summary_file" in results and os.path.exists(results["overall_summary_file"]):
            with open(results["overall_summary_file"], "rb") as file:
                filename = os.path.basename(results["overall_summary_file"])
                st.download_button(
                    label="📄 Download Complete Report",
                    data=file.read(),
                    file_name=filename,
                    mime="text/plain",
                    help="Complete report with metadata, sources, and summary",
                    use_container_width=True
                )
    
    # Additional download option for just the summary text
    if "overall_summary" in results:
        with st.expander("📝 Additional Download Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Simple summary text download
                summary_text = results["overall_summary"]
                st.download_button(
                    label="📄 Download Summary Only",
                    data=summary_text,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    help="Just the summary text without metadata"
                )
            
            with col2:
                # JSON export of all results
                json_data = json.dumps({
                    "document_title": results.get("document_title"),
                    "company_info": results.get("company_info"),
                    "date_range": results.get("date_range"),
                    "top_sources": results.get("top_sources"),
                    "processed_items": results.get("processed_items"),
                    "overall_summary": results.get("overall_summary"),
                    "individual_summaries": results.get("individual_summaries", [])
                }, indent=2)
                
                st.download_button(
                    label="📋 Download JSON Data",
                    data=json_data,
                    file_name=f"news_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="All results in structured JSON format"
                )

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # App header
    st.title("📰 News Summarizer Bot")
    st.markdown("**AI-powered news summarization tool that processes your news data and creates both individual and overall summaries with company insights.**")
    
    st.divider()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key. Get one from https://platform.openai.com/api-keys"
        )
        
        if api_key and not validate_api_key(api_key):
            st.error("Invalid API key format. Should start with 'sk-'")
        
        st.divider()
        
        # File upload section
        st.header("📁 Upload File")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'json'],
            help="Supported formats: CSV, Excel (.xlsx), JSON"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.session_state.uploaded_file_name = uploaded_file.name
        
        st.divider()
        
        # Processing options
        st.header("🎛️ Options")
        
        auto_detect = st.checkbox(
            "Auto-detect columns",
            value=True,
            help="Automatically identify text, source, and date columns"
        )
        
        if uploaded_file and not auto_detect:
            # Get columns from uploaded file
            temp_path = save_uploaded_file(uploaded_file, "temp")
            columns = get_available_columns(temp_path)
            os.remove(temp_path)  # Clean up temp file
            
            if columns:
                selected_column = st.selectbox(
                    "Select text column",
                    options=columns,
                    help="Choose the column containing news text"
                )
                
                if len(columns) > 1:
                    source_column = st.selectbox(
                        "Select source column (optional)",
                        options=["Auto-detect"] + columns,
                        help="Choose the column containing news sources"
                    )
                    
                    date_column = st.selectbox(
                        "Select date column (optional)",
                        options=["Auto-detect"] + columns,
                        help="Choose the column containing dates"
                    )
            else:
                st.error("Could not read file columns")
        
        st.divider()
        
        # About section
        with st.expander("ℹ️ About"):
            st.write("""
            **Enhanced Features:**
            - Auto-detects text, source, and date columns
            - Extracts company name and ticker symbol
            - Identifies top 5 news sources
            - Analyzes date ranges
            - Individual news summarization
            - Overall theme analysis with metadata
            - Professional report generation
            
            **How it works:**
            1. Upload your news file
            2. Bot identifies content and metadata
            3. Extracts company information
            4. Each news item is summarized
            5. Overall summary with insights is generated
            6. Download comprehensive reports
            """)
        
        # Enhanced file format guide
        with st.expander("📋 File Format Guide"):
            st.write("""
            **Required:**
            - At least one column with news text content
            
            **Optional but recommended:**
            - Source/Publisher column (e.g., 'source', 'publisher')
            - Date column (e.g., 'date', 'published')
            - Company/Ticker columns
            
            **Example columns:**
            - text, content, article, news
            - source, publisher, outlet
            - date, timestamp, published
            - company, ticker, symbol
            """)
    
    # Main content area
    if not api_key or not validate_api_key(api_key):
        st.info("👈 Please enter your OpenAI API key in the sidebar to get started.")
        
        with st.expander("🔑 How to get an OpenAI API Key"):
            st.write("""
            1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
            2. Sign up or log in to your account
            3. Navigate to API Keys section
            4. Click "Create new secret key"
            5. Copy the key and paste it in the sidebar
            
            **Note:** You'll need credits in your OpenAI account to use the API.
            """)
        return
    
    if not uploaded_file:
        st.info("👈 Please upload a news file to begin processing.")
        
        # Show enhanced example of expected file format
        with st.expander("📋 Expected File Format Examples"):
            st.write("Your file should contain news text and optionally source/date information:")
            
            example_data = {
                'date': ['2024-01-15', '2024-01-15', '2024-01-16'],
                'source': ['Reuters', 'Bloomberg', 'Financial Times'],
                'headline': ['AAPL reports strong Q4', 'Apple sees growth', 'iPhone sales surge'],
                'text': [
                    'Apple Inc (AAPL) announced strong fourth quarter results with revenue up 15%...',
                    'Apple reported impressive growth in its services division during Q4...',
                    'iPhone sales surged 20% year-over-year, beating analyst expectations...'
                ],
                'company': ['Apple Inc', 'Apple Inc', 'Apple Inc'],
                'ticker': ['AAPL', 'AAPL', 'AAPL']
            }
            
            st.dataframe(pd.DataFrame(example_data))
            st.caption("The bot will automatically extract company info, sources, and dates from your data.")
        
        return
    
    # Process the uploaded file
    if uploaded_file and not st.session_state.processing_complete:
        try:
            # Save uploaded file
            file_path = save_uploaded_file(uploaded_file)
            
            # Load and preview file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(file_path)
            
            # Initialize bot
            bot = NewsSummarizerBot(api_key)
            text_column = bot.identify_text_column(df)
            
            if text_column:
                display_file_preview(df, text_column)
                
                # Show detected metadata
                with st.expander("🔍 Detected Metadata", expanded=False):
                    source_col = bot.identify_source_column(df)
                    date_col = bot.identify_date_column(df)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Text Column:** {text_column}")
                    with col2:
                        st.write(f"**Source Column:** {source_col or 'Not found'}")
                    with col3:
                        st.write(f"**Date Column:** {date_col or 'Not found'}")
                
                # Processing button
                if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                    with st.spinner("Processing your news file..."):
                        results = bot.process_news_file(file_path)
                        st.session_state.results = results
                        st.session_state.processing_complete = True
                        st.rerun()
            else:
                st.error("❌ Could not identify a text column in your file. Please ensure your file contains a column with news text.")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    # Display results if processing is complete
    if st.session_state.processing_complete and st.session_state.results:
        display_results(st.session_state.results)
        
        # Reset button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Process Another File", type="secondary", use_container_width=True):
                st.session_state.processing_complete = False
                st.session_state.results = None
                st.session_state.uploaded_file_name = None
                st.rerun()
        
        with col2:
            if st.button("📤 Share Results", type="secondary", use_container_width=True):
                st.info("💡 Use the download buttons above to save and share your results!")

if __name__ == "__main__":
    main()
