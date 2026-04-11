import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import time

# --- FLIGHT RISK POP-UP ---
@st.dialog("🚨 Risk Analysis - Immediate Action Required", width="large")
def flight_risk_popup(df, active_mask):
    risk_df = df[(active_mask) & (df['Clean_Present_%'] < 70.0)].copy()
    st.markdown("""
        <div style='background-color:#fee2e2; border-left:5px solid #ef4444; padding:20px; border-radius:8px; margin-bottom:20px;'>
            <h4 style='color:#b91c1c; margin:0; font-family:sans-serif;'>High-Risk Fading Students</h4>
            <p style='color:#7f1d1d; margin:5px 0 0 0; font-family:sans-serif; font-size:15px;'>These students are currently marked <b>Active</b> but have fallen below the 70% attendance threshold. They are statistically highly likely to drop out if not contacted immediately.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not risk_df.empty:
        display_cols = ['College', 'Candidate Name']
        if 'Skill' in risk_df.columns: display_cols.append('Skill')
        display_cols.append('Clean_Present_%')
        
        risk_table = risk_df[display_cols].rename(columns={'Clean_Present_%': 'Current Attendance %'})
        risk_table = risk_table.sort_values(by='Current Attendance %', ascending=True)
        risk_table['Current Attendance %'] = risk_table['Current Attendance %'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(risk_table, use_container_width=True, hide_index=True)
    else:
        st.success("✅ Excellent cohort health. No active students currently fall below the 70% attendance threshold.")

# --- HELPER FUNCTIONS ---
def format_percentage(val):
    try:
        if pd.isna(val): return 0.0
        val = str(val).replace('%', '').strip()
        f_val = float(val)
        if f_val <= 1.0: f_val = f_val * 100
        return f_val
    except: return 0.0

def generate_slabs(pct):
    if pct >= 90: return "90-100% (Excellent)"
    elif pct >= 80: return "80-89% (Good)"
    elif pct >= 60: return "60-79% (Warning)"
    else: return "0-59% (Critical)"

def calculate_weekly_trends(df, date_cols):
    if not date_cols: return None, None
    id_vars = ['Superset ID', 'Candidate Name', 'College', 'Standard_Status']
    existing_id_vars = [c for c in id_vars if c in df.columns]
    
    melted = df.melt(id_vars=existing_id_vars, value_vars=date_cols, var_name='Date', value_name='Att_Value')
    melted['Att_Value'] = melted['Att_Value'].astype(str).str.strip().str.upper()
    melted = melted[melted['Att_Value'].isin(['P', 'A'])]
    melted['Date'] = pd.to_datetime(melted['Date'], errors='coerce')
    melted = melted.dropna(subset=['Date'])
    
    if melted.empty: return None, None
    
    melted['Week_Num'] = melted['Date'].dt.isocalendar().week
    melted['Is_Present'] = (melted['Att_Value'] == 'P').astype(int)
    
    min_week = melted['Week_Num'].min()
    melted['Week_Label'] = "Week " + (melted['Week_Num'] - min_week + 1).astype(str)
    
    weekly_trend = melted.groupby(['College', 'Standard_Status', 'Week_Num', 'Week_Label'])['Is_Present'].mean().reset_index()
    weekly_trend['Is_Present'] *= 100
    
    individual_weekly = melted.groupby(['Candidate Name', 'Week_Label'])['Is_Present'].mean().reset_index()
    individual_weekly['Weekly_Status'] = individual_weekly['Is_Present'].apply(lambda x: 'Present' if x >= 0.5 else 'Absent')
    
    return weekly_trend, individual_weekly


def build_weekly_hover_remarks(weekly_frame):
    if weekly_frame is None or weekly_frame.empty:
        return weekly_frame

    weekly_frame = weekly_frame.sort_values("Week_Num").copy()
    remarks = []
    deltas = []
    previous_value = None

    for _, row in weekly_frame.iterrows():
        current_value = float(row["Is_Present"])
        if previous_value is None:
            remarks.append("Opening benchmark week for the current active cohort.")
            deltas.append(0.0)
        else:
            delta = round(current_value - previous_value, 1)
            deltas.append(delta)
            if delta <= -5:
                remarks.append("Sharp dip versus the previous week. Check absentee clusters, schedule disruption, or exam overlap.")
            elif delta < 0:
                remarks.append("Minor dip versus the previous week. Monitor engagement drift and trainer follow-up.")
            elif delta >= 5:
                remarks.append("Strong recovery versus the previous week. Current attendance actions appear to be working.")
            elif delta > 0:
                remarks.append("Slight improvement versus the previous week. Cohort attendance is recovering.")
            else:
                remarks.append("Attendance is stable versus the previous week.")
        previous_value = current_value

    weekly_frame["WoW Delta"] = deltas
    weekly_frame["Week Remark"] = remarks
    return weekly_frame

def process_files(files):
    dfs = []
    for f in files:
        if f is not None:
            df = pd.read_excel(f, sheet_name=0, engine='openpyxl')
            df.columns = df.columns.astype(str).str.strip()
            if 'College' not in df.columns: df['College'] = f.name.replace('.xlsx', '').replace('.xlsm', '')
            dfs.append(df)
            
    if not dfs: return None, None, None
    master_df = pd.concat(dfs, ignore_index=True)
    
    pct_col = next((c for c in master_df.columns if 'present' in c.lower() and '%' in c.lower()), None)
    master_df['Clean_Present_%'] = master_df[pct_col].apply(format_percentage) if pct_col else 0.0
    master_df['Attendance Slab'] = master_df['Clean_Present_%'].apply(generate_slabs)
    
    status_col = next((c for c in master_df.columns if 'status' in c.lower()), None)
    master_df['Standard_Status'] = master_df[status_col].astype(str).str.title().str.strip() if status_col else "Unknown"
    
    skill_col = next((c for c in master_df.columns if 'skill' in c.lower()), None)
    master_df['Skill'] = master_df[skill_col].astype(str).str.strip() if skill_col else "Unknown Skill"

    date_cols = [c for c in master_df.columns if re.match(r'^\d{4}-\d{2}-\d{2}.*', str(c)) or re.match(r'^\d{1,2}-[a-zA-Z]{3}-\d{2}.*', str(c))]
    weekly_df, ind_weekly_df = calculate_weekly_trends(master_df, date_cols)
        
    return master_df, weekly_df, ind_weekly_df

# --- MAIN RUN FUNCTION ---
def run():
    if 'att_data' not in st.session_state: st.session_state['att_data'] = None
    if 'weekly_data' not in st.session_state: st.session_state['weekly_data'] = None
    if 'roster_filter' not in st.session_state: st.session_state['roster_filter'] = 'All'
    if 'scroll_to_roster' not in st.session_state: st.session_state['scroll_to_roster'] = False
    if 'selected_week_drilldown' not in st.session_state: st.session_state['selected_week_drilldown'] = None

    st.markdown(f"""
        <style>
        .portal-header {{ background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); padding: 24px 32px; border-radius: 16px; box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12); display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }}
        .logo-text {{ color: #ffffff; font-weight: 800; font-size: 28px; margin-left: 20px; font-family: 'Segoe UI', sans-serif; }}
        .c-card {{ background: #ffffff; border-radius: 14px; border: 1px solid #e2e8f0; box-shadow: 0 8px 18px rgba(15,23,42,0.06); padding: 20px; display: flex; align-items: center; height: 110px; margin-bottom: 5px; }}
        .c-icon {{ font-size: 26px; width: 55px; height: 55px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; flex-shrink: 0; }}
        .c-text-container {{ display: flex; flex-direction: column; justify-content: center; }}
        .c-value {{ font-size: 28px; font-weight: 800; color: #1e293b; line-height: 1.1; font-family: 'Segoe UI', sans-serif;}}
        .c-label {{ font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}
        .c-trend {{ font-size: 11px; font-weight: bold; margin-top: 3px; }}
        .stButton>button {{ width: 100%; border-radius: 6px; font-weight: 600; height: 35px; background-color: #f8fafc; color: #475569; border: 1px solid #cbd5e1; transition: all 0.2s; font-size: 12px;}}
        .stButton>button:hover {{ background-color: #0f172a; color: #ffffff; border-color: #0f172a; }}
        .sub-label {{ color: #64748b; font-size: 14px; margin-bottom: 15px; display: block; }}
        .upload-shell {{ background:#ffffff; border:1px solid #dbe7f5; border-radius:16px; padding:18px 20px; box-shadow:0 10px 22px rgba(15,23,42,0.05); margin-bottom:18px; }}
        .upload-title {{ color:#0f172a; font-size:20px; font-weight:800; margin-bottom:6px; }}
        .upload-subtitle {{ color:#64748b; font-size:14px; }}
        </style>
        <div class="portal-header">
            <div style="display: flex; align-items: center;">
                <img src="https://img.icons8.com/fluency/96/school-building.png" width="50">
                <div class="logo-text">Executive Attendance Intelligence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state['att_data'] is not None:
        if st.button("🔄 Reset Dashboards / Upload New Files"):
            st.session_state['att_data'] = None
            st.session_state['weekly_data'] = None
            st.session_state['ind_weekly_data'] = None
            st.rerun()

    if st.session_state['att_data'] is None:
        st.markdown("""
            <div class="upload-shell">
                <div class="upload-title">Secure Data Upload</div>
                <div class="upload-subtitle">Upload one or more attendance workbooks for different colleges or cohorts, then generate the executive dashboard in a single ingestion pass.</div>
            </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload Attendance Files",
            type=["xlsx", "xlsm"],
            accept_multiple_files=True,
        )

        if uploaded_files and len(uploaded_files) > 0:
            if st.button("Generate Executive Dashboard", type="primary"):
                with st.spinner("Executing data models and compiling ..."):
                    master, weekly, ind_weekly = process_files(uploaded_files)
                    st.session_state['att_data'] = master
                    st.session_state['weekly_data'] = weekly
                    st.session_state['ind_weekly_data'] = ind_weekly
                    time.sleep(1)
                    st.rerun()
        return
        st.markdown("### 📥 Secure File Upload ")
        
        btn_col1, btn_col2, _ = st.columns([1, 1, 4])
        with btn_col1:
            if st.button("➕ Add Upload Slot"): st.session_state['num_uploaders'] += 1; st.rerun()
        with btn_col2:
            if st.button("➖ Remove Slot") and st.session_state['num_uploaders'] > 1: st.session_state['num_uploaders'] -= 1; st.rerun()
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        uploaded_files = []
        up_cols = st.columns(3)
        for i in range(st.session_state['num_uploaders']):
            with up_cols[i % 3]:
                st.markdown(f"**College File {i+1}**")
                f = st.file_uploader("", type=["xlsx", "xlsm"], key=f"file_{i}", label_visibility="collapsed")
                if f: uploaded_files.append(f)
                
        if len(uploaded_files) > 0:
            if st.button("🚀 Generate Executive Dashboard", type="primary"):
                with st.spinner("Executing data models and compiling ..."):
                    master, weekly, ind_weekly = process_files(uploaded_files)
                    st.session_state['att_data'] = master
                    st.session_state['weekly_data'] = weekly
                    st.session_state['ind_weekly_data'] = ind_weekly
                    time.sleep(1)
                    st.rerun()

    else:
        master_df = st.session_state['att_data']
        weekly_df = st.session_state['weekly_data']
        ind_weekly_df = st.session_state['ind_weekly_data']
        
        colleges = sorted(master_df['College'].unique().tolist())
        st.markdown("### 🎛️ Strategic View")
        col_scope, col_risk = st.columns([3, 1])
        
        with col_scope:
            selected_scope = st.selectbox("Select View Level:", ["Overall BU Performance (All Colleges)"] + colleges, label_visibility="collapsed")
        
        w_df = None
        if selected_scope != "Overall BU Performance (All Colleges)":
            df = master_df[master_df['College'] == selected_scope].copy()
            if weekly_df is not None: w_df = weekly_df[weekly_df['College'] == selected_scope].copy()
        else:
            df = master_df.copy()
            if weekly_df is not None: w_df = weekly_df.copy()
            
        active_mask = df['Standard_Status'].str.contains('Active', case=False, na=False)
        
        with col_risk:
            st.button("🚨 FLIGHT RISK AI - VIEW INSIGHT", on_click=flight_risk_popup, args=(df, active_mask))
                
        st.divider()

        total_students = len(df)
        drop_count = len(df[df['Standard_Status'].str.contains('Drop', case=False, na=False)])
        noshow_mask = df['Standard_Status'].str.contains(r'No Show|Not Join|Dint Join|Didn\'t Join', case=False, na=False)
        noshow_count = len(df[noshow_mask])
        avg_att = df[active_mask]['Clean_Present_%'].mean() if len(df[active_mask]) > 0 else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        
        with m1:
            st.markdown(f"<div class='c-card' style='border-left: 5px solid #3b82f6;'><div class='c-icon' style='color: #3b82f6; background-color: #eff6ff;'>👥</div><div class='c-text-container'><div class='c-value'>{total_students}</div><div class='c-label'>Total Enrolled</div></div></div>", unsafe_allow_html=True)
            if st.button("🔍 Filter All", key="b1"): 
                st.session_state['roster_filter'] = 'All'
                st.session_state['selected_week_drilldown'] = None
                st.session_state['scroll_to_roster'] = True
                st.rerun()
            
        with m2:
            st.markdown(f"<div class='c-card' style='border-left: 5px solid #10b981;'><div class='c-icon' style='color: #10b981; background-color: #d1fae5;'>🎯</div><div class='c-text-container'><div class='c-value'>{len(df[active_mask])}</div><div class='c-label'>Active Cohort</div></div></div>", unsafe_allow_html=True)
            if st.button("🔍 Filter Active", key="b2"): 
                st.session_state['roster_filter'] = 'Active'
                st.session_state['selected_week_drilldown'] = None
                st.session_state['scroll_to_roster'] = True
                st.rerun()

        with m3:
            st.markdown(f"<div class='c-card' style='border-left: 5px solid #ef4444;'><div class='c-icon' style='color: #ef4444; background-color: #fee2e2;'>📉</div><div class='c-text-container'><div class='c-value'>{drop_count}</div><div class='c-label'>Dropouts</div></div></div>", unsafe_allow_html=True)
            if st.button("🔍 Filter Dropouts", key="b3"): 
                st.session_state['roster_filter'] = 'Dropout'
                st.session_state['selected_week_drilldown'] = None
                st.session_state['scroll_to_roster'] = True
                st.rerun()

        with m4:
            st.markdown(f"<div class='c-card' style='border-left: 5px solid #f59e0b;'><div class='c-icon' style='color: #f59e0b; background-color: #fef3c7;'>🚫</div><div class='c-text-container'><div class='c-value'>{noshow_count}</div><div class='c-label'>No Shows</div></div></div>", unsafe_allow_html=True)
            if st.button("🔍 Filter No Shows", key="b4"): 
                st.session_state['roster_filter'] = 'No Show'
                st.session_state['selected_week_drilldown'] = None
                st.session_state['scroll_to_roster'] = True
                st.rerun()

        with m5:
            trend_html = ""
            if w_df is not None and not w_df.empty:
                w_active = w_df[w_df['Standard_Status'].str.contains('Active', case=False, na=False)]
                if not w_active.empty:
                    last_week = w_active.sort_values('Week_Num').iloc[-1]
                    last_week_val = last_week['Is_Present']
                    diff = last_week_val - avg_att
                    color = "#10b981" if diff >= 0 else "#ef4444"
                    arrow = "▲" if diff >= 0 else "▼"
                    trend_html = f"<div class='c-trend' style='color:{color};'>{arrow} Wk: {last_week_val:.1f}%</div>"

            st.markdown(f"<div class='c-card' style='border-left: 5px solid #8b5cf6;'><div class='c-icon' style='color: #8b5cf6; background-color: #ede9fe;'>📊</div><div class='c-text-container'><div class='c-value'>{avg_att:.1f}%</div><div class='c-label'>Avg Active %</div>{trend_html}</div></div>", unsafe_allow_html=True)
            st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if w_df is not None and not w_df.empty:
            w_active = w_df[w_df['Standard_Status'].str.contains('Active', case=False, na=False)]
            if not w_active.empty:
                st.markdown(f"#### 📉 Trajectory & Week-on-Week Trend: {selected_scope}")
                st.markdown("<span class='sub-label'>Click on any specific week's data point on the chart to filter the bottom roster.</span>", unsafe_allow_html=True)
                w_agg = w_active.groupby(['Week_Label', 'Week_Num'])['Is_Present'].mean().reset_index().sort_values('Week_Num')
                w_agg = build_weekly_hover_remarks(w_agg)
                
                if len(w_agg) >= 3:
                    last_3 = w_agg.tail(3)
                    x = np.array([1, 2, 3])
                    y = last_3['Is_Present'].values
                    m, c = np.polyfit(x, y, 1)
                    pred_val = min(100, max(0, (m * 4) + c))
                    
                    next_week_num = w_agg['Week_Num'].iloc[-1] + 1
                    forecast_delta = round(pred_val - float(w_agg['Is_Present'].iloc[-1]), 1)
                    forecast_remark = "AI forecast indicates a likely drop next week; intervene early." if forecast_delta < 0 else "AI forecast suggests stable or improving attendance next week."
                    pred_row = pd.DataFrame({
                        'Week_Label': [f"Week {next_week_num} (AI Forecast)"],
                        'Week_Num': [next_week_num],
                        'Is_Present': [pred_val],
                        'Type': ['Forecast'],
                        'WoW Delta': [forecast_delta],
                        'Week Remark': [forecast_remark],
                    })
                    w_agg['Type'] = 'Actual'
                    w_agg_plot = pd.concat([w_agg, pred_row], ignore_index=True)
                else:
                    w_agg['Type'] = 'Actual'
                    w_agg_plot = w_agg.copy()

                fig_wow = go.Figure()
                actual = w_agg_plot[w_agg_plot['Type'] == 'Actual']
                fig_wow.add_trace(go.Scatter(
                    x=actual['Week_Label'],
                    y=actual['Is_Present'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='#2563eb', width=4),
                    fill='tozeroy',
                    fillcolor='rgba(37, 99, 235, 0.15)',
                    marker=dict(size=12, color='#0f172a', line=dict(width=2, color='white')),
                    customdata=actual[['WoW Delta', 'Week Remark']].to_numpy(),
                    hovertemplate="<b>%{x}</b><br>Active Attendance: %{y:.1f}%<br>WoW Delta: %{customdata[0]:+.1f} pts<br>Remark: %{customdata[1]}<extra></extra>",
                ))
                
                if len(w_agg) >= 3:
                    forecast_line = pd.concat([actual.iloc[-1:], w_agg_plot[w_agg_plot['Type'] == 'Forecast']])
                    fig_wow.add_trace(go.Scatter(
                        x=forecast_line['Week_Label'],
                        y=forecast_line['Is_Present'],
                        mode='lines+markers',
                        name='AI Predicted Momentum',
                        line=dict(color='#8b5cf6', width=4, dash='dash'),
                        marker=dict(size=12, symbol='star', color='#8b5cf6'),
                        customdata=forecast_line[['WoW Delta', 'Week Remark']].to_numpy(),
                        hovertemplate="<b>%{x}</b><br>Active Attendance: %{y:.1f}%<br>WoW Delta: %{customdata[0]:+.1f} pts<br>Remark: %{customdata[1]}<extra></extra>",
                    ))
                
                fig_wow.update_layout(xaxis_title="", yaxis_title="Active Attendance %", yaxis_range=[max(0, w_agg['Is_Present'].min() - 15), 105], plot_bgcolor='white', margin=dict(t=10, b=0, l=0, r=0), height=350, showlegend=True, legend=dict(orientation="h", y=1.1))
                fig_wow.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
                
                try:
                    selected_data = st.plotly_chart(fig_wow, use_container_width=True, on_select="rerun", selection_mode="points")
                    if selected_data and len(selected_data.selection["points"]) > 0:
                        clicked_x = selected_data.selection["points"][0]["x"]
                        if "Forecast" not in clicked_x:
                            st.session_state['selected_week_drilldown'] = clicked_x
                            st.session_state['scroll_to_roster'] = True
                            st.rerun()
                except TypeError:
                    st.plotly_chart(fig_wow, use_container_width=True)
                
        st.divider()

        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.markdown("#### 🎯 Active Cohort Health Distribution")
            slab_counts = df[active_mask]['Attendance Slab'].value_counts().reindex(["90-100% (Excellent)", "80-89% (Good)", "60-79% (Warning)", "0-59% (Critical)"]).fillna(0).reset_index()
            slab_counts.columns = ['Slab', 'Count']
            fig_donut = px.pie(slab_counts, values='Count', names='Slab', hole=0.5, color='Slab', color_discrete_map={"90-100% (Excellent)": "#16a34a", "80-89% (Good)": "#84cc16", "60-79% (Warning)": "#f59e0b", "0-59% (Critical)": "#dc2626"})
            fig_donut.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.2), margin=dict(t=20, b=0, l=0, r=0), height=320)
            st.plotly_chart(fig_donut, use_container_width=True)

        with c2:
            if selected_scope == "Overall BU Performance (All Colleges)":
                st.markdown("#### 🏛️ College Benchmarking")
                col_bench = df[active_mask].groupby('College')['Clean_Present_%'].mean().reset_index().sort_values(by='Clean_Present_%', ascending=True)
                fig_bar = px.bar(col_bench, x='Clean_Present_%', y='College', orientation='h', text='Clean_Present_%', color='Clean_Present_%', color_continuous_scale=['#bfdbfe', '#60a5fa', '#3b82f6', '#2563eb', '#1d4ed8', '#1e3a8a'])
                fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition="outside", cliponaxis=False)
                fig_bar.update_layout(coloraxis_showscale=False, xaxis_title="Avg Active Attendance %", yaxis_title="", plot_bgcolor='white', margin=dict(t=20, b=0, l=0, r=0), height=320)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.markdown(f"#### 🎓 Cohort Benchmarks ({selected_scope})")
                if 'Skill' in df.columns:
                    batch_bench = df[active_mask].groupby('Skill')['Clean_Present_%'].mean().reset_index().sort_values(by='Clean_Present_%', ascending=True)
                    fig_batch = px.bar(batch_bench, x='Clean_Present_%', y='Skill', orientation='h', text='Clean_Present_%', color='Clean_Present_%', color_continuous_scale=['#bfdbfe', '#60a5fa', '#3b82f6', '#2563eb', '#1d4ed8', '#1e3a8a'])
                    fig_batch.update_traces(texttemplate='%{text:.1f}%', textposition="outside", cliponaxis=False)
                    fig_batch.update_layout(coloraxis_showscale=False, xaxis_title="Avg Attendance %", yaxis_title="", plot_bgcolor='white', yaxis=dict(automargin=True), margin=dict(t=10, b=0, l=0, r=0), height=300)
                    st.plotly_chart(fig_batch, use_container_width=True)

        st.divider()

        st.markdown("<div id='roster_target'></div>", unsafe_allow_html=True)
        drill_text = f" | 🔍 Deep Dive: {st.session_state['selected_week_drilldown']}" if st.session_state['selected_week_drilldown'] else ""
        st.markdown(f"### 📋 Executive Roster View - <span style='color:#2563eb;'>Filter: {st.session_state['roster_filter']}{drill_text}</span>", unsafe_allow_html=True)
        
        display_cols = ['College']
        if 'Skill' in df.columns: display_cols.append('Skill')
        display_cols.extend(['Candidate Name', 'Standard_Status'])
        
        days_col = next((c for c in df.columns if 'days' in c.lower() and 'total' in c.lower()), None)
        if days_col: display_cols.append(days_col)
        display_cols.append('Clean_Present_%')

        grid_df = df[[c for c in display_cols if c in df.columns]].copy()
        grid_df.rename(columns={'Clean_Present_%': 'Overall Attendance %'}, inplace=True)
        
        if st.session_state['roster_filter'] == 'Active':
            grid_df = grid_df[grid_df['Standard_Status'].str.contains('Active', case=False, na=False)]
        elif st.session_state['roster_filter'] == 'Dropout':
            grid_df = grid_df[grid_df['Standard_Status'].str.contains('Drop', case=False, na=False)]
        elif st.session_state['roster_filter'] == 'No Show':
            grid_df = grid_df[grid_df['Standard_Status'].str.contains(r'No Show|Not Join|Dint Join|Didn\'t Join', case=False, na=False)]

        if st.session_state['selected_week_drilldown'] and ind_weekly_df is not None:
            target_week = st.session_state['selected_week_drilldown']
            week_data = ind_weekly_df[ind_weekly_df['Week_Label'] == target_week]
            grid_df = pd.merge(grid_df, week_data[['Candidate Name', 'Weekly_Status']], on='Candidate Name', how='left')
            grid_df.rename(columns={'Weekly_Status': f'{target_week} Status'}, inplace=True)

        grid_df['Overall Attendance %'] = grid_df['Overall Attendance %'].apply(lambda x: f"{float(x):.1f}%")
        if days_col in grid_df.columns: grid_df[days_col] = grid_df[days_col].astype(str)

        st.dataframe(grid_df, use_container_width=True, hide_index=True)

        if st.session_state['scroll_to_roster']:
            components.html(
                """
                <script>
                    const doc = window.parent.document;
                    const element = doc.getElementById('roster_target');
                    if (element) { element.scrollIntoView({behavior: 'smooth', block: 'start'}); }
                </script>
                """, height=0
            )
            st.session_state['scroll_to_roster'] = False
