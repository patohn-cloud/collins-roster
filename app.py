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
    
    days_data = []
    for day, sleepovers, longshifts, status in mapping_raw:
        for staff_name in sleepovers:
            days_data.append({
                'date': day,
                'day_num': int(day.split()[1]),
                'staff': staff_name,
                'shift_type': 'Sleepover',
                'status': status
            })
        for staff_name in longshifts:
            days_data.append({
                'date': day,
                'day_num': int(day.split()[1]),
                'staff': staff_name,
                'shift_type': 'Longshift',
                'status': status
            })
    
    shifts_df = pd.DataFrame(days_data)
    
    coverage_matrix = {}
    for day, sleepovers, longshifts, status in mapping_raw:
        all_staff = sleepovers + longshifts
        coverage_matrix[day] = {
            'date': day,
            'sleepovers': sleepovers,
            'longshifts': longshifts,
            'all_staff': all_staff,
            'pairs': list(combinations(all_staff, 2)) if len(all_staff) >= 2 else [],
            'status': status
        }
    
    return staff, shifts_df, coverage_matrix, mapping_raw

class RosterML:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = None
        
    def train(self, df):
        df = df.copy()
        df['is_sleepover'] = (df['shift_type'] == 'Sleepover').astype(int)
        df['is_relief'] = (df['status'] == 'Relief needed').astype(int)
        
        le = LabelEncoder()
        df['staff_encoded'] = le.fit_transform(df['staff'])
        self.label_encoders['staff'] = le
        
        features = ['day_num', 'is_sleepover', 'is_relief', 'staff_encoded']
        self.feature_columns = features
        
        X = df[features]
        y = (df['status'] == 'Match').astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        os.makedirs('models', exist_ok=True)
        with open('models/roster_model.pkl', 'wb') as f:
            pickle.dump(self, f)
        
        return metrics

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
    
    staff_df, shifts_df, coverage_matrix, mapping_raw = load_roster_data()
    
    st.title("📋 Collins Avenue Roster - July 2026")
    
    if user['role'] == 'staff':
        st.info(f"👤 Your personal view: {user['name']}")
        shifts_filtered = shifts_df[shifts_df['staff'] == user['name']]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("😴 Sleepovers")
            sleepovers = shifts_filtered[shifts_filtered['shift_type'] == 'Sleepover']
            for _, row in sleepovers.iterrows():
                emoji = "✅" if row['status'] == 'Match' else "⚠️"
                st.write(f"{emoji} {row['date']}: {row['status']}")
        
        with col2:
            st.subheader("⭐ Longshifts")
            longshifts = shifts_filtered[shifts_filtered['shift_type'] == 'Longshift']
            for _, row in longshifts.iterrows():
                emoji = "✅" if row['status'] == 'Match' else "⚠️"
                st.write(f"{emoji} {row['date']}: {row['status']}")
        
        colleagues = set()
        for _, shift in shifts_filtered.iterrows():
            day_data = coverage_matrix.get(shift['date'], {})
            colleagues.update([s for s in day_data.get('all_staff', []) if s != user['name']])
        
        if colleagues:
            st.subheader("🤝 Your colleagues")
            st.write(", ".join(sorted(colleagues)))
    
    else:
        st.success("👥 Admin view - Full roster")
        
        st.subheader("All Staff Shifts")
        pivot_df = shifts_df.pivot_table(
            index='staff',
            columns='date',
            values='shift_type',
            aggfunc='first',
            fill_value='-'
        )
        st.dataframe(pivot_df)
        
        if st.button("Train ML Model"):
            ml = RosterML()
            with st.spinner("Training..."):
                metrics = ml.train(shifts_df)
                st.success("Model trained!")
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                col2.metric("F1 Score", f"{metrics['f1_score']:.2%}")

if __name__ == "__main__":
    main()
