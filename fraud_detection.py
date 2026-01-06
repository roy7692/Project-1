import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("fraud_detection_pipeline.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'fraud_detection_pipeline.pkl' not found. Please ensure the model file is in the correct directory.")
        return None

model = load_model()

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Fraud Detection System", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .transaction-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
        }
        .payment-card { border-left-color: #4CAF50; }
        .transfer-card { border-left-color: #2196F3; }
        .cashout-card { border-left-color: #FF9800; }
        .deposit-card { border-left-color: #9C27B0; }
        .withdrawal-card { border-left-color: #F44336; }
        .debit-card { border-left-color: #607D8B; }
        
        .fraud-alert {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            animation: pulse 2s infinite;
        }
        .safe-transaction {
            background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
        }
        .suspicious-alert {
            background: linear-gradient(135deg, #ffd93d 0%, #ffa726 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        .metric-positive { color: #51cf66; }
        .metric-negative { color: #ff6b6b; }
        .metric-warning { color: #ffd93d; }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
def initialize_session_state():
    """Initialize all session state variables with proper structure"""
    if 'transaction_history' not in st.session_state:
        st.session_state.transaction_history = []
    
    if 'balance_consistency_stats' not in st.session_state:
        st.session_state.balance_consistency_stats = {
            'total_checked': 0,
            'consistent_transactions': 0,
            'inconsistent_transactions': 0,
            'fraud_detected': 0
        }

# Initialize session state
initialize_session_state()

# --- Transaction Type Configuration ---
TRANSACTION_TYPES = {
    "PAYMENT": {
        "name": "💳 Payment",
        "description": "Direct payment to merchant or service",
        "color": "payment-card",
        "sender_change": "decrease",
        "receiver_change": "increase",
        "icon": "💳"
    },
    "TRANSFER": {
        "name": "🔄 Transfer",
        "description": "Money transfer between accounts",
        "color": "transfer-card",
        "sender_change": "decrease",
        "receiver_change": "increase",
        "icon": "🔄"
    },
    "CASH_OUT": {
        "name": "💰 Cash Out",
        "description": "Withdrawal to physical cash",
        "color": "cashout-card",
        "sender_change": "decrease",
        "receiver_change": "none",
        "icon": "💰"
    },
    "DEPOSIT": {
        "name": "📥 Deposit",
        "description": "Adding funds to account",
        "color": "deposit-card",
        "sender_change": "increase",
        "receiver_change": "none",
        "icon": "📥"
    },
    "WITHDRAWAL": {
        "name": "🏧 Withdrawal",
        "description": "ATM or bank withdrawal",
        "color": "withdrawal-card",
        "sender_change": "decrease",
        "receiver_change": "none",
        "icon": "🏧"
    },
    "DEBIT": {
        "name": "📋 Debit",
        "description": "Direct debit payment",
        "color": "debit-card",
        "sender_change": "decrease",
        "receiver_change": "increase",
        "icon": "📋"
    }
}

# --- Balance Consistency Logic Functions ---
def calculate_expected_balances(transaction_type, amount, oldbalance_org, oldbalance_dest):
    """
    Calculate expected new balances based on transaction type and amount
    """
    tx_config = TRANSACTION_TYPES.get(transaction_type, {})
    
    if tx_config.get('sender_change') == 'decrease':
        expected_new_org = oldbalance_org - amount
    elif tx_config.get('sender_change') == 'increase':
        expected_new_org = oldbalance_org + amount
    else:
        expected_new_org = oldbalance_org
    
    if tx_config.get('receiver_change') == 'increase':
        expected_new_dest = oldbalance_dest + amount
    elif tx_config.get('receiver_change') == 'decrease':
        expected_new_dest = oldbalance_dest - amount
    else:
        expected_new_dest = oldbalance_dest
    
    return expected_new_org, expected_new_dest

def check_balance_consistency(transaction_type, amount, oldbalance_org, newbalance_org, oldbalance_dest, newbalance_dest):
    """
    Check if the actual balances match expected balances after transaction
    """
    inconsistencies = []
    risk_score = 0
    
    # Calculate expected balances
    expected_new_org, expected_new_dest = calculate_expected_balances(
        transaction_type, amount, oldbalance_org, oldbalance_dest
    )
    
    # Check sender balance consistency
    org_tolerance = 0.01
    org_match = abs(newbalance_org - expected_new_org) <= org_tolerance
    
    if not org_match:
        org_diff = abs(newbalance_org - expected_new_org)
        inconsistency_level = "HIGH" if org_diff > amount * 0.1 else "MEDIUM"
        inconsistencies.append(f"Sender balance mismatch: Expected ${expected_new_org:.2f}, Got ${newbalance_org:.2f}")
        risk_score += 40 if inconsistency_level == "HIGH" else 20
    
    # Check receiver balance consistency for applicable transaction types
    tx_config = TRANSACTION_TYPES.get(transaction_type, {})
    if tx_config.get('receiver_change') in ['increase', 'decrease']:
        dest_tolerance = 0.01
        dest_match = abs(newbalance_dest - expected_new_dest) <= dest_tolerance
        
        if not dest_match:
            dest_diff = abs(newbalance_dest - expected_new_dest)
            inconsistency_level = "HIGH" if dest_diff > amount * 0.1 else "MEDIUM"
            inconsistencies.append(f"Receiver balance mismatch: Expected ${expected_new_dest:.2f}, Got ${newbalance_dest:.2f}")
            risk_score += 30 if inconsistency_level == "HIGH" else 15
    
    # Check for insufficient funds for outgoing transactions
    if tx_config.get('sender_change') == 'decrease':
        if amount > oldbalance_org:
            inconsistencies.append(f"Insufficient funds: Attempted ${amount:.2f} with available balance ${oldbalance_org:.2f}")
            risk_score += 50
    
    # Check for negative balances
    if newbalance_org < 0:
        inconsistencies.append(f"Negative sender balance: ${newbalance_org:.2f}")
        risk_score += 30
    
    if newbalance_dest < 0:
        inconsistencies.append(f"Negative receiver balance: ${newbalance_dest:.2f}")
        risk_score += 30
    
    # Check for suspicious balance patterns
    if oldbalance_org == 0 and amount > 1000:
        inconsistencies.append("Large transaction from zero-balance account")
        risk_score += 20
    
    if oldbalance_dest == 0 and amount > 5000:
        inconsistencies.append("Large transaction to new/zero-balance account")
        risk_score += 15
    
    return inconsistencies, risk_score, org_match, expected_new_org, expected_new_dest

def determine_fraud_status(risk_score, inconsistencies):
    """
    Determine if transaction is fraudulent based on risk score and inconsistencies
    """
    if risk_score >= 70:
        return True, "HIGH_RISK"
    elif risk_score >= 40:
        return False, "MEDIUM_RISK"
    else:
        return False, "LOW_RISK"

def safe_increment_stat(stat_key, increment=1):
    """Safely increment a statistic in session state"""
    stats = st.session_state.balance_consistency_stats
    if stat_key in stats:
        stats[stat_key] += increment
    else:
        stats[stat_key] = increment

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🛡️ Fraud Detection System")
    st.info("""
    **Smart Detection Features:**
    - Balance consistency validation
    - Mathematical accuracy checks
    - Real-time risk assessment
    - Multi-mode transaction support
    """)
    
    st.markdown("### 📊 Live Statistics")
    stats = st.session_state.balance_consistency_stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Checked", stats.get('total_checked', 0))
    with col2:
        fraud_count = len([tx for tx in st.session_state.transaction_history if tx.get('is_fraud', False)])
        st.metric("Fraud Detected", fraud_count)
    
    st.markdown("---")
    st.markdown("### ⚙️ System Settings")
    
    balance_tolerance = st.slider(
        "Balance Tolerance ($)", 
        0.01, 10.0, 0.01, 0.01,
        help="Allowed difference between expected and actual balances"
    )

# --- Main Header ---
st.markdown("""
    <div class='main-header'>
        <h1>🛡️ Smart Fraud Detection System</h1>
        <p>Advanced transaction analysis with multi-mode support</p>
    </div>
""", unsafe_allow_html=True)

# --- Main Content in Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Transaction Analysis", 
    "📊 Analytics Dashboard", 
    "📋 Transaction History",
    "ℹ️ System Guide"
])

with tab1:
    st.markdown("### 💰 Multi-Mode Transaction Analysis")
    
    # Transaction Type Selection with Cards
    st.markdown("#### 🎯 Select Transaction Type")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(f"💳 Payment", use_container_width=True):
            st.session_state.selected_type = "PAYMENT"
        if st.button(f"🔄 Transfer", use_container_width=True):
            st.session_state.selected_type = "TRANSFER"
    
    with col2:
        if st.button(f"💰 Cash Out", use_container_width=True):
            st.session_state.selected_type = "CASH_OUT"
        if st.button(f"📥 Deposit", use_container_width=True):
            st.session_state.selected_type = "DEPOSIT"
    
    with col3:
        if st.button(f"🏧 Withdrawal", use_container_width=True):
            st.session_state.selected_type = "WITHDRAWAL"
        if st.button(f"📋 Debit", use_container_width=True):
            st.session_state.selected_type = "DEBIT"
    
    # Default selection
    if 'selected_type' not in st.session_state:
        st.session_state.selected_type = "PAYMENT"
    
    selected_config = TRANSACTION_TYPES[st.session_state.selected_type]
    
    # Transaction Input Form
    with st.container():
        st.markdown(f'<div class="transaction-card {selected_config["color"]}">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"#### {selected_config['icon']} {selected_config['name']}")
            st.markdown(f"*{selected_config['description']}*")
            
            # Amount Input
            amount = st.number_input(
                "**Transaction Amount ($)**", 
                min_value=0.0, 
                value=1000.0, 
                step=100.0, 
                format="%.2f",
                key="amount_input"
            )
        
        with col2:
            # Balance Change Guide
            st.markdown("#### 📈 Expected Changes")
            sender_icon = "📉" if selected_config['sender_change'] == 'decrease' else "📈" if selected_config['sender_change'] == 'increase' else "➡️"
            receiver_icon = "📈" if selected_config['receiver_change'] == 'increase' else "📉" if selected_config['receiver_change'] == 'decrease' else "➡️"
            
            st.markdown(f"{sender_icon} **Sender**: {selected_config['sender_change'].title()}")
            st.markdown(f"{receiver_icon} **Receiver**: {selected_config['receiver_change'].title()}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Balance Input Section
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("#### 👤 Sender Account Details")
            oldbalanceOrg = st.number_input(
                "**Old Balance - Sender ($)**", 
                min_value=0.0, 
                value=10000.0, 
                step=100.0, 
                format="%.2f",
                key="old_sender"
            )
            newbalanceOrig = st.number_input(
                "**New Balance - Sender ($)**", 
                min_value=0.0, 
                value=9000.0, 
                step=100.0, 
                format="%.2f",
                key="new_sender"
            )
    
    with col2:
        with st.container():
            st.markdown("#### 👤 Receiver Account Details")
            oldbalanceDest = st.number_input(
                "**Old Balance - Receiver ($)**", 
                min_value=0.0, 
                value=5000.0, 
                step=100.0, 
                format="%.2f",
                key="old_receiver"
            )
            newbalanceDest = st.number_input(
                "**New Balance - Receiver ($)**", 
                min_value=0.0, 
                value=6000.0, 
                step=100.0, 
                format="%.2f",
                key="new_receiver"
            )

    # --- Real-time Balance Validation Display ---
    st.markdown("### ⚖️ Real-time Validation")
    
    if amount > 0:
        expected_new_org, expected_new_dest = calculate_expected_balances(
            st.session_state.selected_type, amount, oldbalanceOrg, oldbalanceDest
        )
        
        # Validation Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            org_diff = abs(newbalanceOrig - expected_new_org)
            org_status = "✅ MATCH" if org_diff <= 0.01 else "❌ MISMATCH"
            status_class = "metric-positive" if org_diff <= 0.01 else "metric-negative"
            st.metric(
                "Sender Balance", 
                org_status,
                delta=f"${org_diff:.2f}",
                delta_color="normal" if org_diff <= 0.01 else "inverse"
            )
        
        with col2:
            if selected_config['receiver_change'] != 'none':
                dest_diff = abs(newbalanceDest - expected_new_dest)
                dest_status = "✅ MATCH" if dest_diff <= 0.01 else "❌ MISMATCH"
                st.metric(
                    "Receiver Balance", 
                    dest_status,
                    delta=f"${dest_diff:.2f}",
                    delta_color="normal" if dest_diff <= 0.01 else "inverse"
                )
            else:
                st.metric("Receiver Balance", "⚪ N/A")
        
        with col3:
            sufficient_funds = amount <= oldbalanceOrg if selected_config['sender_change'] == 'decrease' else True
            funds_status = "✅ SUFFICIENT" if sufficient_funds else "❌ INSUFFICIENT"
            st.metric(
                "Funds Check", 
                funds_status,
                delta=f"Short: ${amount - oldbalanceOrg:.2f}" if not sufficient_funds else "Adequate",
                delta_color="normal" if sufficient_funds else "inverse"
            )
        
        with col4:
            risk_indicator = "🟢 LOW" if amount < 5000 else "🟡 MEDIUM" if amount < 10000 else "🔴 HIGH"
            st.metric(
                "Amount Risk", 
                risk_indicator,
                delta=f"${amount:,.2f}"
            )

    # --- Analysis Button ---
    st.markdown("---")
    if st.button(f"🚀 Analyze {selected_config['name']} Transaction", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {selected_config['name'].lower()} transaction..."):
            time.sleep(1)
            
            # Update statistics
            safe_increment_stat('total_checked')
            
            # Perform analysis
            inconsistencies, risk_score, org_match, expected_new_org, expected_new_dest = check_balance_consistency(
                st.session_state.selected_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
            )
            
            # Determine fraud status
            is_fraud, risk_level = determine_fraud_status(risk_score, inconsistencies)
            
            # Store transaction record
            tx_record = {
                'id': f"TX{len(st.session_state.transaction_history) + 1:06d}",
                'timestamp': datetime.now(),
                'type': st.session_state.selected_type,
                'type_name': selected_config['name'],
                'amount': amount,
                'sender_old': oldbalanceOrg,
                'sender_new': newbalanceOrig,
                'receiver_old': oldbalanceDest,
                'receiver_new': newbalanceDest,
                'expected_sender': expected_new_org,
                'expected_receiver': expected_new_dest,
                'inconsistencies': inconsistencies,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'is_fraud': is_fraud
            }
            
            # Update statistics
            if not inconsistencies:
                safe_increment_stat('consistent_transactions')
            else:
                safe_increment_stat('inconsistent_transactions')
            
            if is_fraud:
                safe_increment_stat('fraud_detected')
            
            # --- Display Results ---
            st.markdown("## 📋 Analysis Results")
            
            if not inconsistencies:
                with st.container():
                    st.markdown('<div class="safe-transaction">', unsafe_allow_html=True)
                    st.markdown(f"### ✅ {selected_config['icon']} TRANSACTION VALIDATED")
                    st.markdown("**All balance checks passed successfully**")
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                if is_fraud:
                    with st.container():
                        st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
                        st.markdown(f"### 🚨 FRAUD DETECTED - {selected_config['name']}")
                        st.markdown("**Critical inconsistencies identified**")
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    with st.container():
                        st.markdown('<div class="suspicious-alert">', unsafe_allow_html=True)
                        st.markdown(f"### ⚠️ SUSPICIOUS {selected_config['name']}")
                        st.markdown("**Review recommended for this transaction**")
                        st.markdown("</div>", unsafe_allow_html=True)
            
            # Detailed Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Balance comparison chart
                categories = ['Sender Old', 'Sender New', 'Expected Sender']
                values = [oldbalanceOrg, newbalanceOrig, expected_new_org]
                
                if selected_config['receiver_change'] != 'none':
                    categories.extend(['Receiver Old', 'Receiver New', 'Expected Receiver'])
                    values.extend([oldbalanceDest, newbalanceDest, expected_new_dest])
                
                balance_df = pd.DataFrame({
                    'Category': categories,
                    'Amount': values,
                    'Type': ['Actual'] * 2 + ['Expected'] + ['Actual'] * 2 + ['Expected'] if selected_config['receiver_change'] != 'none' else ['Actual'] * 2 + ['Expected']
                })
                
                fig_balances = px.bar(
                    balance_df, x='Category', y='Amount', color='Type',
                    title=f"{selected_config['name']} - Balance Analysis",
                    color_discrete_map={'Actual': '#667eea', 'Expected': '#51cf66'},
                    barmode='group'
                )
                st.plotly_chart(fig_balances, use_container_width=True)
            
            with col2:
                # Risk gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Risk Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70}}
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Issues list
                if inconsistencies:
                    st.markdown("#### 🔍 Issues Identified:")
                    for issue in inconsistencies:
                        st.write(f"• {issue}")
            
            # Final recommendation
            st.markdown("#### 💡 Action Recommendation:")
            if is_fraud:
                st.error(f"**🚨 IMMEDIATE ACTION:** Block this {selected_config['name'].lower()} and alert security team")
            elif risk_level == "MEDIUM_RISK":
                st.warning(f"**⚠️ REVIEW REQUIRED:** Verify this {selected_config['name'].lower()} with additional checks")
            else:
                st.success(f"**✅ APPROVE:** This {selected_config['name'].lower()} appears legitimate")
            
            # Store transaction
            st.session_state.transaction_history.append(tx_record)

with tab2:
    st.markdown("### 📊 Analytics Dashboard")
    
    if st.session_state.transaction_history:
        # Key Metrics
        total_tx = len(st.session_state.transaction_history)
        fraud_tx = len([tx for tx in st.session_state.transaction_history if tx.get('is_fraud', False)])
        legit_tx = total_tx - fraud_tx
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", total_tx)
        with col2:
            st.metric("Fraudulent", fraud_tx)
        with col3:
            st.metric("Legitimate", legit_tx)
        with col4:
            fraud_rate = (fraud_tx / total_tx * 100) if total_tx > 0 else 0
            st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Transaction type distribution
            type_data = pd.Series([tx.get('type_name', 'Unknown') for tx in st.session_state.transaction_history]).value_counts()
            fig_types = px.pie(
                values=type_data.values,
                names=type_data.index,
                title="Transaction Type Distribution",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig_types, use_container_width=True)
        
        with col2:
            # Risk score distribution
            risk_scores = [tx.get('risk_score', 0) for tx in st.session_state.transaction_history]
            fig_risk = px.histogram(
                x=risk_scores, nbins=20,
                title="Risk Score Distribution",
                labels={'x': 'Risk Score', 'y': 'Count'}
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        
        # Fraud by transaction type
        st.markdown("#### 🎯 Fraud Analysis by Transaction Type")
        
        fraud_analysis = {}
        for tx in st.session_state.transaction_history:
            tx_type = tx.get('type_name', 'Unknown')
            if tx_type not in fraud_analysis:
                fraud_analysis[tx_type] = {'total': 0, 'fraud': 0}
            fraud_analysis[tx_type]['total'] += 1
            if tx.get('is_fraud', False):
                fraud_analysis[tx_type]['fraud'] += 1
        
        if fraud_analysis:
            analysis_data = []
            for tx_type, counts in fraud_analysis.items():
                fraud_rate = (counts['fraud'] / counts['total'] * 100) if counts['total'] > 0 else 0
                analysis_data.append({
                    'Transaction Type': tx_type,
                    'Fraud Rate': fraud_rate,
                    'Total': counts['total'],
                    'Fraudulent': counts['fraud']
                })
            
            analysis_df = pd.DataFrame(analysis_data)
            fig_analysis = px.bar(
                analysis_df, x='Transaction Type', y='Fraud Rate',
                title="Fraud Rate by Transaction Type",
                color='Fraud Rate',
                color_continuous_scale='reds'
            )
            st.plotly_chart(fig_analysis, use_container_width=True)
        
    else:
        st.info("No transactions analyzed yet. Start by analyzing transactions in the Transaction Analysis tab.")

with tab3:
    st.markdown("### 📋 Transaction History")
    
    if st.session_state.transaction_history:
        # Display recent transactions
        recent_tx = st.session_state.transaction_history[-10:]
        
        history_data = []
        for tx in recent_tx:
            history_data.append({
                "ID": tx.get('id', 'Unknown'),
                "Time": tx.get('timestamp', datetime.now()).strftime("%H:%M"),
                "Type": tx.get('type_name', 'Unknown'),
                "Amount": f"${tx.get('amount', 0):,.2f}",
                "Risk": tx.get('risk_score', 0),
                "Status": "🔴 Fraud" if tx.get('is_fraud', False) else "🟢 Legit",
                "Issues": len(tx.get('inconsistencies', []))
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)
        
        # Export option
        if st.button("📥 Export Transaction History"):
            export_data = []
            for tx in st.session_state.transaction_history:
                export_data.append({
                    'ID': tx.get('id', 'Unknown'),
                    'Timestamp': tx.get('timestamp', datetime.now()),
                    'Type': tx.get('type_name', 'Unknown'),
                    'Amount': tx.get('amount', 0),
                    'Risk_Score': tx.get('risk_score', 0),
                    'Is_Fraud': tx.get('is_fraud', False),
                    'Issues_Count': len(tx.get('inconsistencies', [])),
                    'Inconsistencies': '; '.join(tx.get('inconsistencies', []))
                })
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"fraud_detection_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No transaction history available.")

with tab4:
    st.markdown("### ℹ️ System Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Supported Transaction Types")
        
        for tx_type, config in TRANSACTION_TYPES.items():
            with st.expander(f"{config['icon']} {config['name']}"):
                st.write(f"**Description**: {config['description']}")
                st.write(f"**Sender**: {config['sender_change'].title()}")
                st.write(f"**Receiver**: {config['receiver_change'].title()}")
    
    with col2:
        st.markdown("#### 🔍 Detection Features")
        
        features = [
            ("Balance Validation", "Mathematical consistency checks"),
            ("Funds Availability", "Sufficient funds verification"),
            ("Pattern Recognition", "Suspicious activity detection"),
            ("Risk Scoring", "Intelligent risk assessment"),
            ("Real-time Analysis", "Instant fraud detection")
        ]
        
        for feature, description in features:
            st.markdown(f"**{feature}**")
            st.markdown(f"<small>{description}</small>", unsafe_allow_html=True)
            st.markdown("---")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🛡️ Smart Fraud Detection System | Multi-Mode Transaction Analysis"
    "</div>", 
    unsafe_allow_html=True
)