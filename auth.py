from __future__ import annotations

import streamlit as st
import base64
from pathlib import Path

from supabase_link import get_supabase_client


def get_base64_of_bin_file(bin_file):
    """Reads a binary file and converts it to base64 for CSS usage."""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None


def check_password() -> bool:
    """Returns True if the user had the correct password."""

    # CRITICAL: Check authentication state FIRST before rendering anything
    # This prevents double form flash during successful login
    if st.session_state.get("password_correct") is True:
        return True

    # --- Load Background Image (Optional) ---
    # Make sure 'login_bg.png' is in the same folder as auth.py
    # If using a different format (jpg), change the mime type below.
    bin_str = get_base64_of_bin_file("login_bg.png")
    
    bg_style = ""
    if bin_str:
        bg_style = f"""
            .stApp {{
                background-image: url("data:image/png;base64,{bin_str}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
        """
    else:
        # Fallback gradient if image is missing
        bg_style = """
            .stApp {
                background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            }
        """

    # --- Inject CSS only once (cached in session state) ---
    if "login_css_injected" not in st.session_state:
        st.markdown(f"""
            <style>
            /* 1. Set Background on the main App container */
            {bg_style}
            
            /* 2. Hide the standard Streamlit header/menu on login screen for immersion */
            header {{visibility: hidden;}}
            [data-testid="stSidebar"] {{display: none;}}
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            
            /* 3. Glassmorphism Login Card */
            [data-testid="stForm"] {{
                max-width: 400px;
                margin: 10vh auto; /* Center vertically roughly */
                padding: 40px;
                
                /* Glass effect */
                background: rgba(15, 23, 42, 0.65); /* Dark semi-transparent */
                backdrop-filter: blur(12px);         /* The "Frosted Glass" blur */
                -webkit-backdrop-filter: blur(12px);
                
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
            }}

            /* 4. Typography & Brand Header */
            .brand-header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .brand-header h1 {{
                color: #ffffff !important;
                font-family: 'Inter', sans-serif;
                font-weight: 700;
                letter-spacing: 1px;
                margin-bottom: 5px;
                font-size: 168px; /* 6× larger (28px * 6) */
                line-height: 1.2;
            }}
            
            /* 5. Input fields styling to match dark theme */
            .stTextInput > div > div {{
                background-color: rgba(255, 255, 255, 0.05);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
            }}
            .stTextInput > div > div:focus-within {{
                border-color: #38bdf8; /* Electric blue focus */
                box-shadow: 0 0 0 1px #38bdf8;
            }}
            /* Input text color - black for readability */
            .stTextInput input {{
                color: #000000 !important;
            }}
            .stTextInput label {{
                color: #cbd5e1 !important;
            }}
            
            /* 6. Button Styling */
            .stButton > button {{
                background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%);
                border: none;
                color: white;
                font-weight: 600;
                padding: 0.6rem 1rem;
                border-radius: 8px;
                transition: all 0.3s ease;
                width: 100%;
            }}
            .stButton > button:hover {{
                box-shadow: 0 0 15px rgba(14, 165, 233, 0.6);
                transform: translateY(-1px);
            }}
            
            /* 7. Error message styling */
            .stAlert {{
                background-color: rgba(239, 68, 68, 0.2);
                border: 1px solid rgba(239, 68, 68, 0.5);
                color: #fca5a5;
                border-radius: 8px;
            }}
            
            /* 8. Spinner styling */
            .stSpinner > div {{
                border-color: #38bdf8 !important;
            }}
            </style>
        """, unsafe_allow_html=True)
        st.session_state["login_css_injected"] = True

    def authenticate() -> None:
        """Check credentials against Supabase - optimized for speed."""
        username = st.session_state.get("username_input", "").strip()
        password = st.session_state.get("password_input", "").strip()
        
        if not username or not password:
            st.session_state["login_error"] = "Please enter both username and password."
            st.rerun()
            return

        try:
            sb = get_supabase_client(prefer_service_role=True)
            # Optimized query: only select needed fields, single query
            res = sb.table("dashboard_users").select("id,username,is_admin").eq("username", username).eq("password", password).limit(1).execute()
            
            if res.data and len(res.data) > 0:
                # Success - set session state immediately
                st.session_state["password_correct"] = True
                st.session_state["user_info"] = res.data[0]
                # Clear sensitive data and error state
                if "password_input" in st.session_state:
                    del st.session_state["password_input"]
                if "username_input" in st.session_state:
                    del st.session_state["username_input"]
                if "login_error" in st.session_state:
                    del st.session_state["login_error"]
                # Force immediate rerun - auth check will return True immediately
                st.rerun()
            else:
                st.session_state["password_correct"] = False
                st.session_state["login_error"] = "Invalid credentials. Please try again."
                st.rerun()
        except Exception as e:
            st.session_state["password_correct"] = False
            st.session_state["login_error"] = f"System Error: {e}"
            st.rerun()

    # --- Login Form UI ---
    
    # 1. Header (Inside the glass card)
    st.markdown("""
        <div class="brand-header">
            <h1>Zelestra Energy</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. Error Message
    if "login_error" in st.session_state:
        st.error(st.session_state["login_error"])
    
    # 3. Form
    with st.form("login", clear_on_submit=False):
        st.text_input("Username", key="username_input", placeholder="Enter username", autocomplete="username")
        st.text_input("Password", type="password", key="password_input", placeholder="••••••••", autocomplete="current-password")
        
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer
        
        submit = st.form_submit_button("Log In", type="primary", use_container_width=True)
        
        if submit:
            with st.spinner("Authenticating..."):
                authenticate()
    
    return False


def logout() -> None:
    """Wipe session state to force login on next refresh."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
