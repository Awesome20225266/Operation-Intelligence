from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from supabase_link import get_supabase_client


@st.cache_data(show_spinner=False)
def get_base64_of_bin_file(bin_file: str):
    """Reads a binary file and converts it to base64 for CSS usage."""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None


def check_password() -> bool:
    """Returns True if the user had the correct password."""

    # CRITICAL: Check authentication state FIRST before rendering ANYTHING
    # This prevents any flash/intermediate screen
    if st.session_state.get("password_correct") is True:
        return True

    # ---------------------------------------------------------------------
    # Boot UX:
    # On initial load (or after logout), users can see a brief blank/dark paint
    # while the login UI initializes. Show an immediate progress bar so the
    # experience is intentional and professional.
    # ---------------------------------------------------------------------
    st.markdown(
        """
        <style>
          html, body { background: #0f172a !important; }
          /* Keep app visible so the boot progress bar can render */
          .stApp { opacity: 1 !important; transition: none !important; }
          header { visibility: hidden; }
          [data-testid="stSidebar"] { display: none !important; }
          #MainMenu { visibility: hidden; }
          footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Immediate boot loader (progress bar) to avoid a blank/dark screen
    import time as _time
    _boot = st.empty()
    with _boot.container():
        st.markdown(
            """
            <div style="
              max-width: 520px;
              margin: 14vh auto 0 auto;
              padding: 18px 20px;
              border-radius: 16px;
              background: rgba(255,255,255,0.06);
              border: 1px solid rgba(255,255,255,0.12);
              color: rgba(255,255,255,0.92);
              font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
            ">
              <div style="font-size:14px; font-weight:700; letter-spacing:0.3px; opacity:0.95;">
                Opening dashboard…
              </div>
              <div style="font-size:12px; opacity:0.78; margin-top:6px;">
                Preparing secure login
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _p_slot = st.empty()
        _p_slot.progress(0, text="Starting… (0%)")
        for i in range(1, 26):
            _p_slot.progress(i, text=f"Starting… ({i}%)")
            _time.sleep(0.01)

    # --- Inject CSS IMMEDIATELY to prevent any unstyled flash ---
    # Load background image (optional)
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

    # --- Always inject CSS first (idempotent, prevents flash) ---
    # This must run BEFORE any other content to prevent unstyled flash
    if "login_css_injected" not in st.session_state:
        st.markdown(f"""
            <style>
            /* 1. Set Background on the main App container */
            {bg_style}
            /* Make login UI visible */
            .stApp {{ opacity: 1 !important; }}
            
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

    # Finish boot loader and unmount it before showing the login form
    try:
        for i in range(26, 101, 4):
            _p_slot.progress(min(100, i), text=f"Loading login… ({min(100, i)}%)")
            _time.sleep(0.01)
    finally:
        _boot.empty()

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
        except ValueError as e:
            # Configuration errors - show clear message
            st.session_state["password_correct"] = False
            error_msg = str(e)
            if "Missing SUPABASE" in error_msg or "Invalid SUPABASE" in error_msg:
                st.session_state["login_error"] = (
                    f"⚠️ **Configuration Error:** {error_msg}\n\n"
                    "Please configure Supabase credentials in `.streamlit/secrets.toml`:\n"
                    "```toml\n"
                    "SUPABASE_URL = 'https://your-project.supabase.co'\n"
                    "SUPABASE_SERVICE_ROLE_KEY = 'your-service-role-key'\n"
                    "```"
                )
            else:
                st.session_state["login_error"] = f"⚠️ **Configuration Error:** {error_msg}"
            st.rerun()
        except ConnectionError as e:
            # Network/connection errors
            st.session_state["password_correct"] = False
            error_msg = str(e)
            st.session_state["login_error"] = (
                f"🔴 **Connection Error:** Cannot connect to Supabase authentication service.\n\n"
                f"{error_msg}\n\n"
                "**Troubleshooting:**\n"
                "1. Check your internet connection\n"
                "2. Verify Supabase URL is correct and accessible\n"
                "3. Check if firewall/proxy is blocking the connection"
            )
            st.rerun()
        except Exception as e:
            # Other unexpected errors
            st.session_state["password_correct"] = False
            error_msg = str(e)
            # Provide helpful message for common errors
            if "getaddrinfo failed" in error_msg or "11001" in error_msg:
                st.session_state["login_error"] = (
                    f"🔴 **Network Error:** Cannot resolve Supabase hostname.\n\n"
                    "**Possible solutions:**\n"
                    "1. Check your internet connection\n"
                    "2. Verify SUPABASE_URL in `.streamlit/secrets.toml` is correct\n"
                    "3. Try accessing Supabase dashboard in browser to confirm service is up\n"
                    "4. Check DNS settings or try using a different network\n\n"
                    f"Technical details: {error_msg}"
                )
            else:
                st.session_state["login_error"] = f"⚠️ **System Error:** {error_msg}"
            st.rerun()

    # --- Login Form UI ---
    
    # 1. Header (Inside the glass card)
    st.markdown("""
        <div class="brand-header">
            <h1>Zel - EYE: OI</h1>
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
        
        submit = st.form_submit_button("Log In", type="primary", width="stretch")
        
        if submit:
            # Progress-first UX (no spinner): mounts immediately, then runs auth
            slot = st.empty()
            prog = slot.progress(0, text="Authenticating... (0%)")
            for p, msg in [(10, "Validating inputs..."), (35, "Checking credentials..."), (70, "Starting session..."), (100, "Done")]:
                prog.progress(p, text=f"{msg} ({p}%)")
            authenticate()
    
    return False


def logout() -> None:
    """Wipe session state to force login on next refresh."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
