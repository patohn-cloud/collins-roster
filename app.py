import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from itertools import combinations

st.set_page_config(page_title="Collins Avenue Roster", page_icon="📋", layout="wide")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_credentials():
    try:
        return pd.read_csv('data/credentials.csv')
    except:
        creds = {
            'name': ['Paul F.', 'Damien F.', 'John G.', 'Davor A.', 'Admin'],
            'password': [hash_password('paul123'), hash_password('damien123'),
                        hash_password('john123'), hash_password('davor123'),
                        hash_password('admin123')],
            'role': ['staff', 'staff', 'staff', 'staff', 'admin']
        }
        df = pd.DataFrame(creds)
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/credentials.csv', index=False)
        return df

def authenticate(username, password):
    creds = load_credentials()
    hashed = hash_password(password)
    user = creds[creds['name'].str.lower() == username.lower()]
    if not user.empty and user.iloc[0]['password'] == hashed:
        return {'authenticated': True, 'name': user.iloc[0]['name'], 'role': user.iloc[0]['role']}
    return {'authenticated': False}

@st.cache_data
def load_roster_data():
    staff = pd.DataFrame({
        'name': ['Paul F.', 'Damien F.', 'John G.', 'Davor A.', 'Martin T.', 'Vlad R.', 'Patricio H.', 'Igor D.'],
        'role': ['SCL', 'SCW', 'DSW', 'DSW', 'SMH', 'SMH', 'Xpress', 'Emerald']
    })
    
    # MAPEO COMPLETO CON TODOS LOS STAFF
    mapping_raw = [
        ('Wed 01', ['John G.'], ['Paul F.'], 'Relief needed'),
        ('Thu 02', ['Martin T.', 'Igor D.'], ['Vlad R.'], 'Match'),
        ('Fri 03', ['Vlad R.'], ['Paul F.'], 'Relief needed'),
        ('Sat 04', ['Martin T.', 'Igor D.'], ['John G.'], 'Match'),
        ('Sun 05', ['John G.', 'Vlad R.'], ['Davor A.'], 'Match'),
        ('Mon 06', ['Paul F.', 'Igor D.'], ['Damien F.'], 'Match'),
        ('Tue 07', ['Damien F.', 'Patricio H.'], ['John G.'], 'Match'),
        ('Wed 08', ['Davor A.', 'Martin T.'], ['Vlad R.'], 'Match'),
        ('Thu 09', ['Damien F.', 'Patricio H.'], ['Paul F.'], 'Match'),
        ('Fri 10', ['Paul F.', 'Vlad R.'], ['Davor A.'], 'Match'),
        ('Sat 11', ['Davor A.', 'Igor D.'], ['Damien F.'], 'Match'),
        ('Sun 12', ['Paul F.', 'Damien F.'], ['Vlad R.'], 'Match'),
        ('Mon 13', ['John G.', 'Vlad R.'], ['Davor A.'], 'Match'),
        ('Tue 14', ['Davor A.'], ['Damien F.', 'Martin T.'], 'Relief needed'),
        ('Wed 15', ['Damien F.', 'Martin T.'], ['Paul F.'], 'Match'),
        ('Thu 16', ['Paul F.', 'Patricio H.'], ['Davor A.'], 'Match'),
        ('Fri 17', ['John G.'], ['Damien F.'], 'Relief needed'),
        ('Sat 18', ['Davor A.', 'Patricio H.'], ['Paul F.'], 'Match'),
        ('Sun 19', ['Damien F.', 'John G.'], [], 'Relief needed'),
        ('Mon 20', ['Davor A.', 'Patricio H.'], ['Paul F.'], 'Match'),
        ('Tue 21', ['Damien F.'], ['John G.'], 'Relief needed'),
        ('Wed 22', ['John G.', 'Patricio H.'], ['Paul F.'], 'Match'),
        ('Thu 23', ['Martin T.', 'Igor D.'], ['Davor A.'], 'Match'),
        ('Fri 24', ['Davor A.', 'Patricio H.'], ['John G.'], 'Match'),
        ('Sat 25', ['John G.', 'Igor D.'], ['Paul F.'], 'Match'),
        ('Sun 26', ['Paul F.', 'Martin T.'], ['Davor A.'], 'Match'),
        ('Mon 27', ['Davor A.', 'Igor D.'], ['John G.'], 'Match'),
        ('Tue 28', ['Martin T.', 'Patricio H.'], ['Paul F.'], 'Match'),
        ('Wed 29', ['Paul F.', 'Igor D.'], ['John G.'], 'Match'),
        ('Thu 30', ['Patricio H.'], ['Davor A.'], 'Relief needed'),
        ('Fri 31', ['Martin T.'], ['Paul F.'], 'Relief needed')
    ]
    
    # Crear diccionario para búsqueda rápida
    day_info = {}
    for day, sleepovers, longshifts, status in mapping_raw:
        day_info[day] = {
            'sleepovers': sleepovers,
            'longshifts': longshifts,
            'status': status,
            'all_staff': sleepovers + longshifts
        }
    
    return staff, day_info, mapping_raw

def main():
    st.sidebar.title("🏢 Collins Avenue")
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['user'] = None
    
    if not st.session_state['logged_in']:
        with st.sidebar.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                auth = authenticate(username, password)
                if auth['authenticated']:
                    st.session_state['logged_in'] = True
                    st.session_state['user'] = auth
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        st.stop()
    
    user = st.session_state['user']
    st.sidebar.success(f"Welcome {user['name']}")
    st.sidebar.write(f"Role: {user['role']}")
    
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['user'] = None
        st.rerun()
    
    staff_df, day_info, mapping_raw = load_roster_data()
    
    st.title("📋 Collins Avenue Roster - July 2026")
    
    if user['role'] == 'staff':
        st.info(f"👤 Your personal view: {user['name']}")
        
        # Encontrar todos los días que trabaja este staff
        my_days = []
        for day, info in day_info.items():
            if user['name'] in info['all_staff']:
                # Determinar qué tipo de turno tiene
                shift_type = None
                if user['name'] in info['sleepovers']:
                    shift_type = 'Sleepover'
                elif user['name'] in info['longshifts']:
                    shift_type = 'Longshift'
                
                my_days.append({
                    'date': day,
                    'shift_type': shift_type,
                    'sleepovers': info['sleepovers'],
                    'longshifts': info['longshifts'],
                    'status': info['status'],
                    'all_staff': info['all_staff']
                })
        
        # Ordenar por día
        my_days = sorted(my_days, key=lambda x: int(x['date'].split()[1]))
        
        # Separar sleepovers y longshifts
        sleepovers = [d for d in my_days if d['shift_type'] == 'Sleepover']
        longshifts = [d for d in my_days if d['shift_type'] == 'Longshift']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("😴 Sleepovers (2 Required)")
            if sleepovers:
                for day in sleepovers:
                    emoji = "✅" if day['status'] == 'Match' else "⚠️"
                    st.markdown(f"""
                    <div style='background: #e3f2fd; padding: 10px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #1976d2;'>
                        <b>{emoji} {day['date']}</b><br>
                        <b>Sleepovers:</b> {', '.join(day['sleepovers'])}<br>
                        <b>Longshift:</b> {', '.join(day['longshifts']) if day['longshifts'] else 'None'}<br>
                        <b>Status:</b> {day['status']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No sleepovers assigned")
        
        with col2:
            st.subheader("⭐ Longshifts (1 Required)")
            if longshifts:
                for day in longshifts:
                    emoji = "✅" if day['status'] == 'Match' else "⚠️"
                    st.markdown(f"""
                    <div style='background: #fce4ec; padding: 10px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #c62828;'>
                        <b>{emoji} {day['date']}</b><br>
                        <b>Sleepovers:</b> {', '.join(day['sleepovers'])}<br>
                        <b>Longshift:</b> {', '.join(day['longshifts']) if day['longshifts'] else 'None'}<br>
                        <b>Status:</b> {day['status']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No longshifts assigned")
        
        # Resumen de compañeros
        st.subheader("🤝 Todos tus compañeros de turno")
        all_coworkers = set()
        for day in my_days:
            for s in day['all_staff']:
                if s != user['name']:
                    all_coworkers.add(s)
        
        if all_coworkers:
            st.success("Trabajas con: " + ", ".join(sorted(all_coworkers)))
        else:
            st.info("Trabajas solo")
    
    else:
        # Admin view - Roster completo
        st.success("👥 Admin view - Full roster")
        
        # Mostrar el mapeo completo
        st.subheader("📅 Daily Coverage Mapping - Complete Roster")
        for day, info in day_info.items():
            emoji = "✅" if info['status'] == 'Match' else "⚠️"
            with st.expander(f"{emoji} {day} - {info['status']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**😴 Sleepovers (2 Required):**")
                    for s in info['sleepovers']:
                        st.write(f"- {s}")
                    if len(info['sleepovers']) < 2:
                        st.warning(f"⚠️ Need {2 - len(info['sleepovers'])} more sleepover(s)")
                
                with col2:
                    st.write("**⭐ Longshift (1 Required):**")
                    if info['longshifts']:
                        for s in info['longshifts']:
                            st.write(f"- {s}")
                    else:
                        st.warning("⚠️ No longshift assigned")
                        st.write("**Need 1 longshift!**")
        
        # Estadísticas
        st.subheader("📊 Roster Statistics")
        total_days = len(day_info)
        match_days = sum(1 for d in day_info.values() if d['status'] == 'Match')
        relief_days = total_days - match_days
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Days", total_days)
        col2.metric("Match", match_days)
        col3.metric("Relief Needed", relief_days)
        col4.metric("Coverage Rate", f"{match_days/total_days:.1%}")

if __name__ == "__main__":
    main()
