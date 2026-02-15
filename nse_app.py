"""
NSE Google Trends Analyzer - All-in-One Streamlit App
Analyzes 2000+ NSE stocks with parallel processing
"""

import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import yfinance as yf
from datetime import datetime
import time
from scipy import stats
from multiprocessing import Pool, cpu_count
import requests
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_WORKERS = max(1, cpu_count() - 1)  # Parallel workers
BATCH_SIZE = 50  # Stocks per batch
NSE_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def simplify_company_name(name):
    """Simplify company name for Google Trends"""
    name = str(name).upper()
    
    # Remove common suffixes
    suffixes = [' LIMITED', ' LTD', ' LTD.', ' INDIA', ' PVT', ' PRIVATE', 
                ' CORPORATION', ' CORP', ' COMPANY', ' CO', ' INDUSTRIES', ' IND']
    
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:len(name)-len(suffix)]
            break
    
    # Take first 2-3 words if too long
    words = name.strip().split()
    if len(words) > 3:
        name = ' '.join(words[:2])
    
    return name.strip()


def fetch_nse_universe():
    """Fetch all NSE stocks"""
    try:
        df = pd.read_csv(NSE_URL)
        df.columns = df.columns.str.strip()
        
        # Filter for equity series
        if 'SERIES' in df.columns:
            df = df[df['SERIES'].str.strip() == 'EQ']
        
        # Process
        df['TICKER'] = df['SYMBOL'].str.strip() + '.NS'
        df['COMPANY_NAME'] = df['NAME OF COMPANY'].str.strip()
        df['SEARCH_TERM'] = df['COMPANY_NAME'].apply(simplify_company_name)
        
        return df[['TICKER', 'SYMBOL', 'COMPANY_NAME', 'SEARCH_TERM']].copy()
    except Exception as e:
        st.error(f"Error fetching NSE universe: {e}")
        return None


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_single_stock(args):
    """Analyze one stock (worker function)"""
    ticker, search_term = args
    
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        price_data = stock.history(period='1y')
        
        if price_data is None or len(price_data) < 60:
            return None
        
        current_price = price_data['Close'].iloc[-1]
        price_3m_ago = price_data['Close'].iloc[min(-63, -len(price_data))]
        price_change_3m = ((current_price / price_3m_ago) - 1) * 100
        
        # Calculate price slope
        recent = price_data['Close'].tail(63)
        x = np.arange(len(recent))
        y = recent.values
        price_slope, _, _, _, _ = stats.linregress(x, y)
        price_slope_norm = (price_slope / recent.iloc[0]) * 100
        
        # Get Google Trends data
        time.sleep(2)  # Rate limiting
        pytrends = TrendReq(hl='en-US', tz=360)
        
        try:
            pytrends.build_payload([search_term], cat=0, timeframe='today 12-m', geo='', gprop='')
            trends_data = pytrends.interest_over_time()
            
            if trends_data is not None and not trends_data.empty:
                if 'isPartial' in trends_data.columns:
                    trends_data = trends_data.drop('isPartial', axis=1)
                
                if search_term in trends_data.columns:
                    trends_series = trends_data[search_term]
                    
                    # Calculate trends slope
                    recent_trends = trends_series.tail(12)
                    x = np.arange(len(recent_trends))
                    y = recent_trends.values
                    
                    if np.sum(y) > 0:
                        trends_slope, _, _, _, _ = stats.linregress(x, y)
                        mean_val = recent_trends.mean()
                        trends_slope_norm = (trends_slope / mean_val * 100) if mean_val > 0 else 0
                        
                        # Search percentile
                        current_interest = trends_series.iloc[-1]
                        search_percentile = stats.percentileofscore(trends_series.values, current_interest)
                        
                        # Calculate conviction score
                        divergence = trends_slope_norm - price_slope_norm
                        divergence_score = ((np.clip(divergence, -50, 50) + 50) / 100) * 100
                        
                        conviction = (
                            divergence_score * 0.40 +
                            search_percentile * 0.40 +
                            50 * 0.20
                        )
                        
                        return {
                            'Ticker': ticker,
                            'Company': search_term,
                            'Current Price': round(current_price, 2),
                            'Price Change 3M (%)': round(price_change_3m, 2),
                            'Price Slope': round(price_slope_norm, 4),
                            'Trends Slope': round(trends_slope_norm, 4),
                            'Divergence': round(divergence, 4),
                            'Search Percentile': round(search_percentile, 2),
                            'Conviction Score': round(conviction, 2),
                            'Status': 'Complete'
                        }
        except:
            pass
        
        # If trends failed, return price-only data
        return {
            'Ticker': ticker,
            'Company': search_term,
            'Current Price': round(current_price, 2),
            'Price Change 3M (%)': round(price_change_3m, 2),
            'Price Slope': round(price_slope_norm, 4),
            'Trends Slope': None,
            'Divergence': None,
            'Search Percentile': None,
            'Conviction Score': None,
            'Status': 'Price Only'
        }
        
    except Exception as e:
        return None


def analyze_batch_parallel(stocks_df, num_workers=None):
    """Analyze stocks in parallel"""
    if num_workers is None:
        num_workers = NUM_WORKERS
    
    # Prepare work items
    work_items = [(row['TICKER'], row['SEARCH_TERM']) for _, row in stocks_df.iterrows()]
    
    # Process in parallel
    results = []
    with Pool(processes=num_workers) as pool:
        for result in pool.imap(analyze_single_stock, work_items):
            if result:
                results.append(result)
    
    return pd.DataFrame(results)


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="NSE Trends Analyzer", layout="wide")
    
    st.title("📊 NSE Google Trends Analyzer")
    st.markdown("Analyze 2000+ NSE stocks using Google Trends + Price Data")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Quick Test (50 stocks)", "Nifty 500", "Top 1000", "Full Universe (2000+)"]
    )
    
    # Map mode to stock count
    stock_limits = {
        "Quick Test (50 stocks)": 50,
        "Nifty 500": 500,
        "Top 1000": 1000,
        "Full Universe (2000+)": None
    }
    max_stocks = stock_limits[analysis_mode]
    
    num_workers = st.sidebar.slider(
        "Parallel Workers",
        min_value=1,
        max_value=cpu_count(),
        value=max(1, cpu_count() - 1),
        help=f"Your system has {cpu_count()} CPU cores"
    )
    
    st.sidebar.info(f"""
    **System Info:**
    - CPU Cores: {cpu_count()}
    - Workers: {num_workers}
    - Expected Speedup: ~{num_workers}x
    """)
    
    # Main area
    if 'universe' not in st.session_state:
        st.session_state.universe = None
    
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Step 1: Fetch Universe
    st.header("Step 1: Fetch NSE Universe")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🔄 Fetch NSE Stocks", type="primary"):
            with st.spinner("Fetching NSE stock universe..."):
                universe = fetch_nse_universe()
                if universe is not None:
                    st.session_state.universe = universe
                    st.success(f"✓ Fetched {len(universe)} stocks!")
                else:
                    st.error("Failed to fetch universe")
    
    if st.session_state.universe is not None:
        with col2:
            st.metric("Total NSE Stocks", len(st.session_state.universe))
        
        # Show sample
        with st.expander("📋 Sample Stocks"):
            st.dataframe(st.session_state.universe.head(20), use_container_width=True)
    
    # Step 2: Analyze
    if st.session_state.universe is not None:
        st.header("Step 2: Analyze Stocks")
        
        stocks_to_analyze = max_stocks if max_stocks else len(st.session_state.universe)
        
        st.info(f"""
        **Selected:** {analysis_mode}
        - Stocks to analyze: {stocks_to_analyze}
        - Workers: {num_workers}
        - Estimated time: {stocks_to_analyze * 3 / num_workers / 60:.0f}-{stocks_to_analyze * 5 / num_workers / 60:.0f} minutes
        """)
        
        if st.button("▶️ Start Analysis", type="primary"):
            # Prepare stocks
            stocks = st.session_state.universe.head(max_stocks) if max_stocks else st.session_state.universe
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            
            # Analyze in batches for progress updates
            all_results = []
            total_batches = (len(stocks) + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_num in range(total_batches):
                batch_start = batch_num * BATCH_SIZE
                batch_end = min(batch_start + BATCH_SIZE, len(stocks))
                batch = stocks.iloc[batch_start:batch_end]
                
                status_text.text(f"Processing batch {batch_num + 1}/{total_batches} ({batch_start + 1}-{batch_end} of {len(stocks)})")
                
                # Analyze batch
                batch_results = analyze_batch_parallel(batch, num_workers)
                
                if not batch_results.empty:
                    all_results.append(batch_results)
                
                # Update progress
                progress = (batch_end) / len(stocks)
                progress_bar.progress(progress)
            
            # Combine results
            if all_results:
                results_df = pd.concat(all_results, ignore_index=True)
                st.session_state.results = results_df
                
                elapsed = time.time() - start_time
                
                st.success(f"""
                ✓ Analysis Complete!
                - Stocks analyzed: {len(results_df)}
                - Time taken: {elapsed/60:.1f} minutes
                - Speed: {len(results_df) / elapsed * 60:.1f} stocks/minute
                """)
            else:
                st.error("No results generated")
    
    # Step 3: View Results
    if st.session_state.results is not None:
        st.header("Step 3: Results & Insights")
        
        results = st.session_state.results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        complete = results[results['Status'] == 'Complete']
        
        with col1:
            st.metric("Total Analyzed", len(results))
        
        with col2:
            st.metric("Complete Analysis", len(complete))
        
        with col3:
            if len(complete) > 0:
                avg_conviction = complete['Conviction Score'].mean()
                st.metric("Avg Conviction", f"{avg_conviction:.1f}")
        
        with col4:
            if len(complete) > 0:
                high_conviction = len(complete[complete['Conviction Score'] > 70])
                st.metric("High Conviction (>70)", high_conviction)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Top Opportunities", "📈 All Results", "💎 Watchlists", "📥 Download"])
        
        with tab1:
            st.subheader("Top 20 Conviction Scores")
            
            if len(complete) > 0:
                top_20 = complete.nlargest(20, 'Conviction Score')
                
                # Color code by conviction
                def color_conviction(val):
                    if pd.isna(val):
                        return ''
                    if val >= 75:
                        return 'background-color: #90EE90'
                    elif val >= 60:
                        return 'background-color: #FFE4B5'
                    elif val >= 40:
                        return 'background-color: #E0E0E0'
                    else:
                        return 'background-color: #FFB6C1'
                
                styled = top_20.style.applymap(color_conviction, subset=['Conviction Score'])
                st.dataframe(styled, use_container_width=True)
            else:
                st.warning("No complete analysis results available")
        
        with tab2:
            st.subheader("All Results")
            
            # Filters
            col1, col2 = st.columns(2)
            
            with col1:
                status_filter = st.multiselect(
                    "Status",
                    options=results['Status'].unique(),
                    default=results['Status'].unique()
                )
            
            with col2:
                if 'Conviction Score' in results.columns:
                    min_conviction = st.slider("Min Conviction Score", 0, 100, 0)
                else:
                    min_conviction = 0
            
            # Filter
            filtered = results[results['Status'].isin(status_filter)]
            if 'Conviction Score' in filtered.columns:
                filtered = filtered[
                    (filtered['Conviction Score'].isna()) | 
                    (filtered['Conviction Score'] >= min_conviction)
                ]
            
            st.dataframe(filtered, use_container_width=True)
        
        with tab3:
            st.subheader("Automated Watchlists")
            
            if len(complete) > 0:
                # Top Conviction
                st.markdown("**🏆 Top Conviction (>70)**")
                top_conv = complete[complete['Conviction Score'] > 70].nlargest(50, 'Conviction Score')
                st.dataframe(top_conv[['Ticker', 'Company', 'Conviction Score', 'Divergence']], use_container_width=True)
                
                # High Divergence
                st.markdown("**💎 High Divergence (Value Plays)**")
                high_div = complete[complete['Divergence'] > 5].nlargest(50, 'Divergence')
                st.dataframe(high_div[['Ticker', 'Company', 'Divergence', 'Conviction Score']], use_container_width=True)
                
                # Momentum
                st.markdown("**🚀 Momentum Plays**")
                momentum = complete[
                    (complete['Price Change 3M (%)'] > 10) & 
                    (complete['Search Percentile'] > 70)
                ].nlargest(50, 'Price Change 3M (%)')
                st.dataframe(momentum[['Ticker', 'Company', 'Price Change 3M (%)', 'Search Percentile']], use_container_width=True)
            else:
                st.warning("No watchlists available - need complete analysis data")
        
        with tab4:
            st.subheader("Download Results")
            
            # CSV download
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv,
                file_name=f"nse_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
            # Top conviction only
            if len(complete) > 0:
                top_csv = complete.nlargest(100, 'Conviction Score').to_csv(index=False)
                st.download_button(
                    label="📥 Download Top 100 (CSV)",
                    data=top_csv,
                    file_name=f"nse_top100_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **How to interpret:**
    - **Conviction Score 70-100:** 🟢 Strong Buy Signal
    - **Conviction Score 60-69:** 🟡 Moderate Buy
    - **Conviction Score 40-59:** ⚪ Neutral
    - **Positive Divergence:** Search interest rising faster than price (bullish)
    - **Negative Divergence:** Price rising faster than search (bearish)
    
    *Not financial advice. Use as one input among many for investment decisions.*
    """)


if __name__ == "__main__":
    main()
