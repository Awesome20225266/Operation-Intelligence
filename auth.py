from __future__ import annotations

import streamlit as st

from supabase_link import get_supabase_client


def check_password() -> bool:
    """Returns True if the user had the correct password."""

    # CRITICAL: Check authentication state FIRST before rendering anything
    # This prevents double form flash during successful login
    if st.session_state.get("password_correct") is True:
        return True

    # Inject CSS only once (cached in session state)
    if "login_css_injected" not in st.session_state:
        st.markdown("""
            <style>
            /* Center the login container */
            [data-testid="stForm"] {
                max-width: 400px;
                margin: 80px auto;
                border: 1px solid #e6e9ef;
                border-radius: 15px;
                padding: 30px;
                background-color: white;
                box-shadow: 0 10px 25px rgba(0,0,0,0.05);
            }
            
            /* Brand Header */
            .brand-header {
                text-align: center;
                margin-bottom: 30px;
            }
            .brand-header h1 {
                color: #1E3A8A; /* Dark Blue */
                font-weight: 800;
                font-size: 26px;
                margin-bottom: 5px;
            }
            .brand-header p {
                color: #6B7280;
                font-size: 14px;
            }

            /* Hide Streamlit elements during login */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
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

    # Show Login Form (for both first visit and failed attempts)
    st.markdown("""
        <div class="brand-header">
            <h1>Zelestra Energy</h1>
            <p>Enter your credentials to access the portal</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Show error message if any
    if "login_error" in st.session_state:
        st.error(st.session_state["login_error"])
    
    with st.form("login", clear_on_submit=False):
        st.text_input("Username", key="username_input", placeholder="e.g. shashank_k", autocomplete="username")
        st.text_input("Password", type="password", key="password_input", placeholder="••••••••", autocomplete="current-password")
        submit = st.form_submit_button("Login to Dashboard", type="primary", use_container_width=True)
        if submit:
            with st.spinner("Authenticating..."):
                authenticate()
    
    return False


def logout() -> None:
    """Wipe session state to force login on next refresh."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
