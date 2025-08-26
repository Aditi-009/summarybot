import streamlit as st
import pandas as pd
import os
import io
from pathlib import Path
import json
from datetime import datetime
import sys
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import re

# Import the backend module
from backend import NewsSummarizerBot, get_available_columns

# Configure Streamlit page
st.set_page_config(
    page_title="News and Social Media Summarizer Bot",
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
    if 'email_sent' not in st.session_state:
        st.session_state.email_sent = False

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    return api_key.startswith('sk-') and len(api_key) > 20

def send_results_via_email(email_address: str, results: dict, sender_email: str, sender_password: str) -> bool:
    """Send summarization results via email"""
    try:
        # Email configuration
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        
        # Create message container
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email_address
        msg['Subject'] = f"📰 {results.get('document_title', 'Content Summary Report')}"
        
        # Create HTML email body
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(90deg, #1f77b4, #2e8b57); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #1f77b4; background-color: #f8f9fa; }}
                .summary-box {{ background-color: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 15px 0; }}
                .meta-info {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 15px 0; }}
                .meta-item {{ background: #e9ecef; padding: 10px; border-radius: 5px; min-width: 150px; }}
                .sources {{ background: #f1f3f4; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{results.get('document_title', 'Content Summary Report')}</h1>
                <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Summary Details</h2>
                <div class="meta-info">
        """
        
        # Add company information
        company_info = results.get("company_info", {})
        if company_info.get('company_name'):
            html_body += f'<div class="meta-item"><strong>Company:</strong> {company_info["company_name"]}</div>'
        if company_info.get('ticker'):
            html_body += f'<div class="meta-item"><strong>Ticker:</strong> {company_info["ticker"]}</div>'
        
        # Add date range
        date_range = results.get("date_range", {})
        if date_range.get('start_date'):
            if date_range['start_date'] == date_range.get('end_date', ''):
                html_body += f'<div class="meta-item"><strong>Date:</strong> {date_range["start_date"]}</div>'
            else:
                html_body += f'<div class="meta-item"><strong>Date Range:</strong> {date_range["start_date"]} to {date_range.get("end_date", "")}</div>'
        
        html_body += f'<div class="meta-item"><strong>Items Processed:</strong> {results.get("processed_items", 0)}</div>'
        html_body += '</div></div>'
        
        # Add top sources
        top_sources = results.get("top_sources", [])
        if top_sources:
            html_body += '''
            <div class="section">
                <h2>📈 Top Content Sources</h2>
                <div class="sources">
            '''
            for i, (source, count) in enumerate(top_sources, 1):
                html_body += f'<p><strong>{i}. {source}</strong> - {count} items</p>'
            html_body += '</div></div>'
        
        # Add overall summary
        html_body += f'''
        <div class="section">
            <h2>📋 Overall Summary</h2>
            <div class="summary-box">
                {results.get("overall_summary", "No summary available").replace(chr(10), "<br>")}
            </div>
        </div>
        '''
        
        # Add footer
        html_body += '''
        <div class="section" style="text-align: center; color: #666;">
            <p>This summary was generated automatically by News and Social Media Summarizer Bot</p>
            <p>📧 Results delivered to your inbox | 🤖 Powered by AI</p>
        </div>
        </body>
        </html>
        '''
        
        # Attach HTML body
        msg.attach(MIMEText(html_body, 'html'))
        
        # Attach files if they exist
        if "overall_summary_file" in results and os.path.exists(results["overall_summary_file"]):
            with open(results["overall_summary_file"], "rb") as f:
                attachment = MIMEApplication(f.read(), _subtype="txt")
                attachment.add_header('Content-Disposition', 'attachment', 
                                    filename=os.path.basename(results["overall_summary_file"]))
                msg.attach(attachment)
        
        if "output_file" in results and os.path.exists(results["output_file"]):
            with open(results["output_file"], "rb") as f:
                attachment = MIMEApplication(f.read(), _subtype="csv")
                attachment.add_header('Content-Disposition', 'attachment', 
                                    filename=os.path.basename(results["output_file"]))
                msg.attach(attachment)
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        return True
        
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

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
        st.info("🔄 Processing your content file...")
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
                status_text.text("Summarizing individual content items...")
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
    """Display top content sources"""
    top_sources = results.get("top_sources", [])
    
    if top_sources:
        st.subheader("📊 Top Content Sources")
        
        # Create a more visual display of sources
        for i, (source, count) in enumerate(top_sources, 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{i}. {source}**")
            with col2:
                st.metric("Items", count)
        
        st.divider()

def display_results(results: dict):
    """Display processing results"""
    if results.get("error"):
        st.error(f"❌ Error: {results['error']}")
        return
    
    st.success("✅ Processing completed successfully!")
    
    # Display document title prominently
    document_title = results.get("document_title", "Content Summary Report")
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
    
    # Display top content sources
    display_sources_info(results)
    
    # Display overall summary
    st.subheader("📋 Overall Content Summary")
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
    
    # Email delivery section
    st.subheader("📧 Email Delivery")
    
    # Email configuration section
    with st.expander("⚙️ Email Configuration (Required for sending)", expanded=not st.session_state.email_sent):
        st.markdown("""
        **To send emails, you need to configure SMTP settings:**
        
        📋 **For Gmail users:**
        1. Enable 2-Factor Authentication on your Google account
        2. Generate an App Password: [Google App Passwords](https://myaccount.google.com/apppasswords)
        3. Use your Gmail address and the generated App Password below
        
        📋 **For other email providers:**
        - Use your email provider's SMTP settings
        """)
        
        email_col1, email_col2 = st.columns(2)
        
        with email_col1:
            sender_email = st.text_input(
                "📧 Your Email Address (sender):",
                placeholder="your-email@gmail.com",
                help="The email address that will send the results"
            )
        
        with email_col2:
            sender_password = st.text_input(
                "🔑 Email App Password:",
                type="password",
                placeholder="App Password (not your regular password)",
                help="For Gmail: Use App Password, not your regular password"
            )
    
    if not st.session_state.email_sent:
        if sender_email and sender_password:
            email_col1, email_col2 = st.columns([2, 1])
            
            with email_col1:
                user_email = st.text_input(
                    "📬 Recipient Email Address:",
                    placeholder="recipient@example.com",
                    help="Where to send the summary results"
                )
            
            with email_col2:
                st.write("")  # Spacer
                st.write("")  # Spacer
                if st.button("📨 Send Results", type="primary", disabled=not user_email or not validate_email(user_email)):
                    if validate_email(user_email):
                        with st.spinner("Sending email..."):
                            if send_results_via_email(user_email, results, sender_email, sender_password):
                                st.success(f"✅ Results sent successfully to {user_email}!")
                                st.session_state.email_sent = True
                                st.rerun()
                            else:
                                st.error("❌ Failed to send email. Please check your credentials and try again.")
                    else:
                        st.error("Please enter a valid recipient email address")
        else:
            st.info("👆 Please configure your email settings above to send results via email.")
    else:
        st.success("✅ Results have been sent via email!")
        if st.button("📧 Send to Different Email", type="secondary"):
            st.session_state.email_sent = False
            st.rerun()
    
    # Display individual summaries
    with st.expander("🔍 View Individual Summaries", expanded=False):
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
        with st.expander("📎 Additional Download Options"):
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
                    file_name=f"content_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="All results in structured JSON format"
                )

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # App header
    st.title("📰 News and Social Media Summarizer Bot")
    st.markdown("**AI-powered content summarization tool that processes news articles, social media posts, and other text content to create both individual and overall summaries with company insights.**")
    
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
                    help="Choose the column containing text content"
                )
                
                if len(columns) > 1:
                    source_column = st.selectbox(
                        "Select source column (optional)",
                        options=["Auto-detect"] + columns,
                        help="Choose the column containing content sources"
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
            - Identifies top 5 content sources
            - Analyzes date ranges
            - Individual content summarization
            - Overall theme analysis with metadata
            - Professional report generation
            - **📧 Email delivery of results**
            
            **How it works:**
            1. Upload your content file (news, social media, etc.)
            2. Bot identifies content and metadata
            3. Extracts company information
            4. Each content item is summarized
            5. Overall summary with insights is generated
            6. Results are emailed to you automatically
            7. Download comprehensive reports
            """)
        
        # Enhanced file format guide
        with st.expander("📋 File Format Guide"):
            st.write("""
            **Required:**
            - At least one column with text content
            
            **Optional but recommended:**
            - Source/Publisher column (e.g., 'source', 'publisher')
            - Date column (e.g., 'date', 'published')
            - Company/Ticker columns
            
            **Example columns:**
            - text, content, article, news, post, comment
            - source, publisher, outlet, platform
            - date, timestamp, published, created_at
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
        st.info("👈 Please upload a content file to begin processing.")
        
        # Show enhanced example of expected file format
        with st.expander("📋 Expected File Format Examples"):
            st.write("Your file should contain text content and optionally source/date information:")
            
            example_data = {
                'date': ['2024-01-15', '2024-01-15', '2024-01-16'],
                'source': ['Reuters', 'Reddit', 'Financial Times'],
                'headline': ['AAPL reports strong Q4', 'Apple sees growth discussion', 'iPhone sales surge'],
                'text': [
                    'Apple Inc (AAPL) announced strong fourth quarter results with revenue up 15%...',
                    'User discussion on r/investing: Apple reported impressive growth in its services division during Q4...',
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
                    with st.spinner("Processing your content file..."):
                        results = bot.process_news_file(file_path)
                        st.session_state.results = results
                        st.session_state.processing_complete = True
                        st.rerun()
            else:
                st.error("❌ Could not identify a text column in your file. Please ensure your file contains a column with text content.")
                
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
                st.session_state.email_sent = False
                st.rerun()
        
        with col2:
            if st.button("📤 Share Results", type="secondary", use_container_width=True):
                st.info("💡 Enter your email above to receive results in your inbox!")

if __name__ == "__main__":
    main()