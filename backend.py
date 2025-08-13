import pandas as pd
import json
import openai
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import os
from pathlib import Path
import time
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsSummarizerBot:
    def __init__(self, api_key: str):
        """Initialize the news summarizer bot with OpenAI API key"""
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        
    def load_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from various file formats"""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension == '.xlsx':
                df = pd.read_excel(file_path)
            elif file_extension == '.json':
                df = pd.read_json(file_path)
            else:
                logger.error(f"Unsupported file format: {file_extension}")
                return None
                
            logger.info(f"Successfully loaded file with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            return None
    
    def identify_text_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify the text column from the dataframe"""
        # Common text column names
        text_column_candidates = [
            'text', 'content', 'news', 'article', 'description', 
            'summary', 'body', 'message', 'title', 'headline'
        ]
        
        # Check for exact matches first
        for col in df.columns:
            if col.lower() in text_column_candidates:
                return col
        
        # Check for partial matches
        for col in df.columns:
            for candidate in text_column_candidates:
                if candidate in col.lower():
                    return col
        
        # If no match found, find the column with longest average text length
        text_lengths = {}
        for col in df.columns:
            if df[col].dtype == 'object':  # String columns
                avg_length = df[col].astype(str).str.len().mean()
                text_lengths[col] = avg_length
        
        if text_lengths:
            return max(text_lengths, key=text_lengths.get)
        
        logger.warning("Could not identify text column automatically")
        return None
    
    def identify_source_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify the source/publisher column from the dataframe"""
        source_column_candidates = [
            'source', 'publisher', 'publication', 'outlet', 'provider',
            'news_source', 'media_source', 'author', 'site', 'website'
        ]
        
        # Check for exact matches first
        for col in df.columns:
            if col.lower() in source_column_candidates:
                return col
        
        # Check for partial matches
        for col in df.columns:
            for candidate in source_column_candidates:
                if candidate in col.lower():
                    return col
        
        logger.info("No source column found")
        return None
    
    def identify_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify the date column from the dataframe"""
        date_column_candidates = [
            'date', 'time', 'timestamp', 'published', 'created',
            'pub_date', 'publish_date', 'datetime', 'created_at'
        ]
        
        # Check for exact matches first
        for col in df.columns:
            if col.lower() in date_column_candidates:
                return col
        
        # Check for partial matches
        for col in df.columns:
            for candidate in date_column_candidates:
                if candidate in col.lower():
                    return col
        
        # Check for datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        
        logger.info("No date column found")
        return None
    
    def get_company_name_from_ticker(self, ticker: str) -> str:
        """Map ticker symbols to proper company names"""
        ticker_to_company = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'GOOG': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'BRK.A': 'Berkshire Hathaway Inc.',
            'BRK.B': 'Berkshire Hathaway Inc.',
            'UNH': 'UnitedHealth Group Inc.',
            'JNJ': 'Johnson & Johnson',
            'XOM': 'Exxon Mobil Corporation',
            'JPM': 'JPMorgan Chase & Co.',
            'V': 'Visa Inc.',
            'PG': 'Procter & Gamble Co.',
            'MA': 'Mastercard Inc.',
            'HD': 'Home Depot Inc.',
            'CVX': 'Chevron Corporation',
            'LLY': 'Eli Lilly and Company',
            'ABBV': 'AbbVie Inc.',
            'BAC': 'Bank of America Corp.',
            'AVGO': 'Broadcom Inc.',
            'KO': 'Coca-Cola Co.',
            'WMT': 'Walmart Inc.',
            'COST': 'Costco Wholesale Corp.',
            'PEP': 'PepsiCo Inc.',
            'TMO': 'Thermo Fisher Scientific Inc.',
            'MRK': 'Merck & Co. Inc.',
            'ADBE': 'Adobe Inc.',
            'NFLX': 'Netflix Inc.',
            'DIS': 'Walt Disney Co.',
            'ABT': 'Abbott Laboratories',
            'ACN': 'Accenture Plc',
            'CRM': 'Salesforce Inc.',
            'NKE': 'Nike Inc.',
            'TXN': 'Texas Instruments Inc.',
            'QCOM': 'QUALCOMM Inc.',
            'VZ': 'Verizon Communications Inc.',
            'CMCSA': 'Comcast Corp.',
            'DHR': 'Danaher Corp.',
            'NEE': 'NextEra Energy Inc.',
            'INTC': 'Intel Corp.',
            'WFC': 'Wells Fargo & Co.',
            'IBM': 'International Business Machines Corp.',
            'AMD': 'Advanced Micro Devices Inc.',
            'T': 'AT&T Inc.',
            'COP': 'ConocoPhillips',
            'UNP': 'Union Pacific Corp.',
            'HON': 'Honeywell International Inc.',
            'RTX': 'RTX Corp.',
            'PM': 'Philip Morris International Inc.',
            'SPGI': 'S&P Global Inc.',
            'CAT': 'Caterpillar Inc.',
            'GS': 'Goldman Sachs Group Inc.',
            'SCHW': 'Charles Schwab Corp.',
            'AXP': 'American Express Co.',
            'NOW': 'ServiceNow Inc.',
            'ISRG': 'Intuitive Surgical Inc.',
            'BLK': 'BlackRock Inc.',
            'SYK': 'Stryker Corp.',
            'BKNG': 'Booking Holdings Inc.',
            'TJX': 'TJX Companies Inc.',
            'ADP': 'Automatic Data Processing Inc.',
            'GILD': 'Gilead Sciences Inc.',
            'MDLZ': 'Mondelez International Inc.',
            'VRTX': 'Vertex Pharmaceuticals Inc.',
            'MMC': 'Marsh & McLennan Companies Inc.',
            'C': 'Citigroup Inc.',
            'LRCX': 'Lam Research Corp.',
            'ZTS': 'Zoetis Inc.',
            'REGN': 'Regeneron Pharmaceuticals Inc.',
            'CB': 'Chubb Ltd.',
            'PGR': 'Progressive Corp.',
            'TMUS': 'T-Mobile US Inc.',
            'SO': 'Southern Co.',
            'BSX': 'Boston Scientific Corp.',
            'SHW': 'Sherwin-Williams Co.',
            'ETN': 'Eaton Corp. Plc',
            'MU': 'Micron Technology Inc.',
            'DUK': 'Duke Energy Corp.',
            'EQIX': 'Equinix Inc.',
            'AON': 'Aon Plc',
            'APD': 'Air Products and Chemicals Inc.',
            'ICE': 'Intercontinental Exchange Inc.',
            'CL': 'Colgate-Palmolive Co.',
            'CSX': 'CSX Corp.',
            'CME': 'CME Group Inc.',
            'USB': 'U.S. Bancorp',
            'ECL': 'Ecolab Inc.',
            'NSC': 'Norfolk Southern Corp.',
            'ITW': 'Illinois Tool Works Inc.',
            'FDX': 'FedEx Corp.',
            'WM': 'Waste Management Inc.',
            'GD': 'General Dynamics Corp.',
            'EOG': 'EOG Resources Inc.',
            'FCX': 'Freeport-McMoRan Inc.',
            'PYPL': 'PayPal Holdings Inc.',
            'PANW': 'Palo Alto Networks Inc.',
            'EL': 'Estee Lauder Companies Inc.',
            'PSA': 'Public Storage',
            'GM': 'General Motors Co.',
            'F': 'Ford Motor Co.',
            'RIVN': 'Rivian Automotive Inc.',
            'LCID': 'Lucid Group Inc.',
            'NIO': 'NIO Inc.',
            'XPEV': 'XPeng Inc.',
            'LI': 'Li Auto Inc.',
            # Add more as needed
        }
        
        return ticker_to_company.get(ticker.upper(), '')

    def clean_source_name(self, source: str) -> str:
        """Clean and extract source name from URL or full name"""
        if pd.isna(source) or not source:
            return ""
        
        source = str(source).strip()
        
        # If it's a URL, extract the domain
        if source.startswith(('http://', 'https://', 'www.')):
            try:
                from urllib.parse import urlparse
                parsed = urlparse(source if source.startswith('http') else 'http://' + source)
                domain = parsed.netloc.lower()
                
                # Remove www. prefix
                if domain.startswith('www.'):
                    domain = domain[4:]
                
                # Map common domains to clean names
                domain_mapping = {
                    'reuters.com': 'Reuters',
                    'bloomberg.com': 'Bloomberg',
                    'wsj.com': 'Wall Street Journal',
                    'ft.com': 'Financial Times',
                    'cnbc.com': 'CNBC',
                    'cnn.com': 'CNN',
                    'bbc.com': 'BBC',
                    'marketwatch.com': 'MarketWatch',
                    'yahoo.com': 'Yahoo Finance',
                    'finance.yahoo.com': 'Yahoo Finance',
                    'fool.com': 'Motley Fool',
                    'seekingalpha.com': 'Seeking Alpha',
                    'benzinga.com': 'Benzinga',
                    'zacks.com': 'Zacks',
                    'morningstar.com': 'Morningstar',
                    'barrons.com': 'Barrons',
                    'investopedia.com': 'Investopedia',
                    'thestreet.com': 'TheStreet',
                    'forbes.com': 'Forbes',
                    'fortune.com': 'Fortune',
                    'businessinsider.com': 'Business Insider',
                    'techcrunch.com': 'TechCrunch',
                    'venturebeat.com': 'VentureBeat',
                    'theverge.com': 'The Verge',
                    'arstechnica.com': 'Ars Technica',
                    'engadget.com': 'Engadget',
                    'wired.com': 'Wired',
                    'npr.org': 'NPR',
                    'apnews.com': 'Associated Press',
                    'ap.org': 'Associated Press',
                    'usatoday.com': 'USA Today',
                    'nytimes.com': 'New York Times',
                    'washingtonpost.com': 'Washington Post',
                    'guardian.co.uk': 'The Guardian',
                    'theguardian.com': 'The Guardian',
                    'economist.com': 'The Economist',
                    'axios.com': 'Axios',
                    'politico.com': 'Politico',
                    'sec.gov': 'SEC',
                    'edgar.sec.gov': 'SEC EDGAR'
                }
                
                clean_name = domain_mapping.get(domain)
                if clean_name:
                    return clean_name
                
                # If not in mapping, try to create a clean name from domain
                # Remove common TLDs and format nicely
                name_part = domain.split('.')[0]
                if name_part in ['finance', 'money', 'news', 'business']:
                    # Handle cases like finance.yahoo.com
                    parts = domain.split('.')
                    if len(parts) > 1:
                        name_part = parts[1]
                
                return name_part.title()
                
            except Exception:
                # If URL parsing fails, continue with original source
                pass
        
        # Clean up common source name patterns
        source = source.replace('www.', '').replace('.com', '').replace('http://', '').replace('https://', '')
        
        # Handle common news source formats
        source_mapping = {
            'wsj': 'Wall Street Journal',
            'ft': 'Financial Times',
            'nyt': 'New York Times',
            'wapo': 'Washington Post',
            'wp': 'Washington Post',
            'lat': 'Los Angeles Times',
            'usat': 'USA Today',
            'ap': 'Associated Press',
            'reuters': 'Reuters',
            'bloomberg': 'Bloomberg',
            'cnbc': 'CNBC',
            'cnn': 'CNN',
            'bbc': 'BBC',
            'npr': 'NPR',
            'abc': 'ABC News',
            'cbs': 'CBS News',
            'nbc': 'NBC News',
            'fox': 'Fox News',
            'mw': 'MarketWatch',
            'yf': 'Yahoo Finance',
            'sa': 'Seeking Alpha',
            'tmf': 'Motley Fool',
            'bi': 'Business Insider',
            'tc': 'TechCrunch',
            'vb': 'VentureBeat'
        }
        
        source_lower = source.lower().strip()
        clean_name = source_mapping.get(source_lower)
        if clean_name:
            return clean_name
        
        # If no mapping found, title case the source
        return source.title()

    def extract_company_info(self, df: pd.DataFrame, text_column: str) -> Dict[str, str]:
        """Extract company name and ticker from the dataset"""
        # Look for company and ticker in column names first
        company_info = {'company_name': '', 'ticker': ''}
        
        # Check column names for company/ticker info
        for col in df.columns:
            col_lower = col.lower()
            if 'company' in col_lower or 'firm' in col_lower or 'corp' in col_lower:
                if not df[col].isna().all():
                    company_info['company_name'] = str(df[col].iloc[0])
            elif 'ticker' in col_lower or 'symbol' in col_lower:
                if not df[col].isna().all():
                    company_info['ticker'] = str(df[col].iloc[0]).upper()
        
        # If ticker found, get proper company name
        if company_info['ticker'] and not company_info['company_name']:
            mapped_name = self.get_company_name_from_ticker(company_info['ticker'])
            if mapped_name:
                company_info['company_name'] = mapped_name
        
        # If not found in columns, try to extract from text content using AI
        if not company_info['company_name'] or not company_info['ticker']:
            sample_texts = df[text_column].dropna().head(3).tolist()
            if sample_texts:
                combined_sample = ' '.join([str(text)[:500] for text in sample_texts])
                extracted_info = self.extract_company_from_text(combined_sample)
                
                if not company_info['ticker'] and extracted_info['ticker']:
                    company_info['ticker'] = extracted_info['ticker'].upper()
                    # Get proper company name from ticker
                    mapped_name = self.get_company_name_from_ticker(company_info['ticker'])
                    if mapped_name:
                        company_info['company_name'] = mapped_name
                
                if not company_info['company_name'] and extracted_info['company_name']:
                    company_info['company_name'] = extracted_info['company_name']
        
        return company_info
    
    def extract_company_from_text(self, text: str) -> Dict[str, str]:
        """Use AI to extract company name and ticker from text"""
        prompt = f"""
        From the following news text, extract the main company name and stock ticker symbol (if mentioned).
        Return only the company name and ticker, or "Not found" if not clearly identifiable.
        
        Text: {text}
        
        Please respond in this exact format:
        Company: [company name or "Not found"]
        Ticker: [ticker symbol or "Not found"]
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting company information from financial news text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse the response
            company_name = "Not found"
            ticker = "Not found"
            
            for line in content.split('\n'):
                if line.strip().startswith('Company:'):
                    company_name = line.split(':', 1)[1].strip()
                elif line.strip().startswith('Ticker:'):
                    ticker = line.split(':', 1)[1].strip()
            
            return {
                'company_name': company_name if company_name != "Not found" else "",
                'ticker': ticker if ticker != "Not found" else ""
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract company info from text: {str(e)}")
            return {'company_name': '', 'ticker': ''}
    
    def get_top_sources(self, df: pd.DataFrame, source_column: str, top_n: int = 5) -> List[Tuple[str, int]]:
        """Get top N news sources from the dataset with cleaned names"""
        if source_column not in df.columns:
            return []
        
        # Clean all source names
        cleaned_sources = df[source_column].dropna().apply(self.clean_source_name)
        
        # Remove empty sources
        cleaned_sources = cleaned_sources[cleaned_sources != ""]
        
        # Count occurrences of each cleaned source
        source_counts = cleaned_sources.value_counts().head(top_n)
        return [(source, count) for source, count in source_counts.items()]
    
    def get_date_range(self, df: pd.DataFrame, date_column: str) -> Dict[str, str]:
        """Get date range from the dataset"""
        if date_column not in df.columns:
            return {'start_date': '', 'end_date': ''}
        
        try:
            # Convert to datetime if not already
            dates = pd.to_datetime(df[date_column], errors='coerce').dropna()
            
            if dates.empty:
                return {'start_date': '', 'end_date': ''}
            
            start_date = dates.min().strftime('%Y-%m-%d')
            end_date = dates.max().strftime('%Y-%m-%d')
            
            return {'start_date': start_date, 'end_date': end_date}
            
        except Exception as e:
            logger.warning(f"Failed to extract date range: {str(e)}")
            return {'start_date': '', 'end_date': ''}
    
    def summarize_single_text(self, text: str, max_retries: int = 3) -> str:
        """Summarize a single news item using GPT-4"""
        if not text or pd.isna(text):
            return "No content to summarize"
        
        # Clean the text
        text = str(text).strip()
        if len(text) < 50:  # Skip very short texts
            return text
        
        prompt = f"""
        Please provide a concise summary of the following news article in 2-3 sentences, focusing on the key facts and implications:
        
        {text}
        
        Summary:
        """
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a professional news summarizer. Provide clear, concise, and factual summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.3
                )
                
                summary = response.choices[0].message.content.strip()
                logger.info(f"Successfully summarized text (attempt {attempt + 1})")
                return summary
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to summarize after {max_retries} attempts")
                    return f"Error summarizing: {str(e)}"
    
    def create_overall_summary(self, summaries: List[str], metadata: Dict[str, Any]) -> str:
        """Create an overall summary from individual summaries with enhanced metadata"""
        if not summaries:
            return "No summaries to process"
        
        # Filter out error messages and empty summaries
        valid_summaries = [s for s in summaries if s and not s.startswith("Error") and s != "No content to summarize"]
        
        if not valid_summaries:
            return "No valid summaries found"
        
        combined_text = " ".join(valid_summaries)
        
        # Build context from metadata
        context_parts = []
        if metadata.get('company_name'):
            context_parts.append(f"Company: {metadata['company_name']}")
        if metadata.get('ticker'):
            context_parts.append(f"Ticker: {metadata['ticker']}")
        if metadata.get('date_range', {}).get('start_date'):
            date_range = metadata['date_range']
            if date_range['start_date'] == date_range['end_date']:
                context_parts.append(f"Date: {date_range['start_date']}")
            else:
                context_parts.append(f"Date Range: {date_range['start_date']} to {date_range['end_date']}")
        
        context = " | ".join(context_parts) if context_parts else ""
        
        prompt = f"""
        Based on the following individual news summaries{f' about {context}' if context else ''}, create a comprehensive overview in 2-3 paragraphs that captures the main themes, trends, and key developments:
        
        {combined_text}
        
        Please provide:
        1. A brief overview of the main topics/themes
        2. Key developments and their implications
        3. Any notable trends or patterns
        
        Overall Summary:
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior news analyst. Provide comprehensive yet concise analysis of multiple news items, identifying patterns and key themes."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            overall_summary = response.choices[0].message.content.strip()
            logger.info("Successfully created overall summary")
            return overall_summary
            
        except Exception as e:
            logger.error(f"Failed to create overall summary: {str(e)}")
            return f"Error creating overall summary: {str(e)}"
    
    def create_document_title(self, metadata: Dict[str, Any]) -> str:
        """Create a formatted document title with company info, ticker, date and item count"""
        title_parts = ["News Summary"]
        
        if metadata.get('company_name'):
            title_parts.append(f"- {metadata['company_name']}")
        
        if metadata.get('ticker'):
            title_parts.append(f"({metadata['ticker']})")
        
        date_range = metadata.get('date_range', {})
        if date_range.get('start_date'):
            if date_range['start_date'] == date_range.get('end_date', ''):
                title_parts.append(f"- {date_range['start_date']}")
            else:
                title_parts.append(f"- {date_range['start_date']} to {date_range.get('end_date', '')}")
        
        if metadata.get('processed_items'):
            title_parts.append(f"- {metadata['processed_items']} News Items")
        
        return " ".join(title_parts)
    
    def format_sources_section(self, top_sources: List[Tuple[str, int]]) -> str:
        """Format the top sources section"""
        if not top_sources:
            return "No source information available."
        
        sources_text = "Top News Sources:\n"
        for i, (source, count) in enumerate(top_sources, 1):
            sources_text += f"{i}. {source} ({count} articles)\n"
        
        return sources_text
    
    def process_news_file(self, file_path: str, output_dir: str = "output") -> Dict[str, Any]:
        """Main method to process news file and generate summaries"""
        logger.info(f"Starting processing of file: {file_path}")
        
        # Load the file
        df = self.load_file(file_path)
        if df is None:
            return {"error": "Failed to load file"}
        
        # Identify columns
        text_column = self.identify_text_column(df)
        if text_column is None:
            return {"error": "Could not identify text column"}
        
        source_column = self.identify_source_column(df)
        date_column = self.identify_date_column(df)
        
        logger.info(f"Using column '{text_column}' as text source")
        if source_column:
            logger.info(f"Using column '{source_column}' as source")
        if date_column:
            logger.info(f"Using column '{date_column}' as date")
        
        # Extract metadata
        company_info = self.extract_company_info(df, text_column)
        top_sources = self.get_top_sources(df, source_column) if source_column else []
        date_range = self.get_date_range(df, date_column) if date_column else {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Summarize individual texts
        summaries = []
        original_texts = df[text_column].tolist()
        
        logger.info(f"Processing {len(original_texts)} news items...")
        
        for i, text in enumerate(original_texts):
            logger.info(f"Processing item {i+1}/{len(original_texts)}")
            summary = self.summarize_single_text(text)
            summaries.append(summary)
            time.sleep(0.5)  # Rate limiting
        
        # Prepare metadata for overall summary
        metadata = {
            'company_name': company_info['company_name'],
            'ticker': company_info['ticker'],
            'date_range': date_range,
            'processed_items': len(summaries),
            'top_sources': top_sources
        }
        
        # Create overall summary
        overall_summary = self.create_overall_summary(summaries, metadata)
        
        # Create document title
        document_title = self.create_document_title(metadata)
        
        # Create results dataframe
        results_df = df.copy()
        results_df['individual_summary'] = summaries
        
        # Save individual summaries to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename based on company info
        filename_parts = ["news_summaries"]
        if company_info['company_name']:
            clean_name = "".join(c for c in company_info['company_name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename_parts.append(clean_name.replace(' ', '_'))
        if company_info['ticker']:
            filename_parts.append(company_info['ticker'])
        filename_parts.append(timestamp)
        
        output_filename = f"{'_'.join(filename_parts)}.csv"
        output_path = os.path.join(output_dir, output_filename)
        results_df.to_csv(output_path, index=False)
        
        # Create enhanced summary document
        overall_summary_filename = f"{'_'.join(filename_parts[:-1])}_summary_{timestamp}.txt"
        overall_summary_path = os.path.join(output_dir, overall_summary_filename)
        
        with open(overall_summary_path, 'w', encoding='utf-8') as f:
            f.write(f"{document_title}\n")
            f.write("=" * len(document_title) + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Add metadata section
            f.write("SUMMARY DETAILS\n")
            f.write("-" * 20 + "\n")
            if company_info['company_name']:
                f.write(f"Company: {company_info['company_name']}\n")
            if company_info['ticker']:
                f.write(f"Ticker Symbol: {company_info['ticker']}\n")
            if date_range.get('start_date'):
                if date_range['start_date'] == date_range.get('end_date', ''):
                    f.write(f"Date: {date_range['start_date']}\n")
                else:
                    f.write(f"Date Range: {date_range['start_date']} to {date_range.get('end_date', '')}\n")
            f.write(f"Total News Items: {len(summaries)}\n")
            
            # Add top sources
            if top_sources:
                f.write(f"\n{self.format_sources_section(top_sources)}")
            
            f.write(f"\nOVERALL SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(overall_summary)
        
        result = {
            "success": True,
            "processed_items": len(summaries),
            "text_column_used": text_column,
            "source_column_used": source_column,
            "date_column_used": date_column,
            "company_info": company_info,
            "top_sources": top_sources,
            "date_range": date_range,
            "document_title": document_title,
            "output_file": output_path,
            "overall_summary": overall_summary,
            "overall_summary_file": overall_summary_path,
            "individual_summaries": summaries
        }
        
        logger.info(f"Processing completed successfully. Output saved to: {output_path}")
        return result

def get_available_columns(file_path: str) -> List[str]:
    """Helper function to get available columns from a file"""
    try:
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(file_path, nrows=1)
        elif file_extension == '.xlsx':
            df = pd.read_excel(file_path, nrows=1)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        else:
            return []
            
        return df.columns.tolist()
    except Exception:
        return []

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    api_key = "your-openai-api-key-here"  # Replace with your actual API key
    bot = NewsSummarizerBot(api_key)
    
    # Test with a sample file
    # result = bot.process_news_file("sample_news.csv")
    # print(json.dumps(result, indent=2))