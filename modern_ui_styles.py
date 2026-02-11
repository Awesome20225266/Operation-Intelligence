"""
Modern UI Styles for S1, S2, S3 Portals
Professional design system with smooth animations and clean aesthetics
"""

MODERN_UI_CSS = """
<style>
/* ========================================================================
   GLOBAL DESIGN SYSTEM
   ======================================================================== */

/* Modern Color Palette */
:root {
    --primary-blue: #2563eb;
    --primary-blue-dark: #1e40af;
    --primary-blue-light: #3b82f6;
    /* Light-blue surfaces (for readable dark text) */
    --primary-blue-50: #eff6ff;
    --primary-blue-100: #dbeafe;
    --primary-blue-200: #bfdbfe;
    --success-green: #10b981;
    --success-green-dark: #059669;
    --warning-orange: #f97316;
    --danger-red: #ef4444;
    --closed-green: #065f46;
    --neutral-50: #f8fafc;
    --neutral-100: #f1f5f9;
    --neutral-200: #e2e8f0;
    --neutral-300: #cbd5e1;
    --neutral-400: #94a3b8;
    --neutral-500: #64748b;
    --neutral-600: #475569;
    --neutral-700: #334155;
    --neutral-800: #1e293b;
    --neutral-900: #0f172a;
}

/* ========================================================================
   SMOOTH PAGE TRANSITIONS & LOADING
   ======================================================================== */

/* Fade in main content */
.main .block-container {
    animation: fadeInContent 0.4s ease-out;
}

@keyframes fadeInContent {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Remove Streamlit default margins for cleaner look */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* ========================================================================
   MODERN NAVIGATION TABS (st.tabs)
   ======================================================================== */

/* Tab container */
.stTabs {
    margin-top: 1rem;
    margin-bottom: 2rem;
}

/* Tab list container */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    padding: 0.5rem 0.25rem;
    border-bottom: 2px solid var(--neutral-200);
    background: transparent;
}

/* Individual tabs */
.stTabs [data-baseweb="tab"] {
    background: white !important;
    border: 1.5px solid var(--neutral-200) !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    margin: 0 !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    color: var(--neutral-700) !important;
    letter-spacing: -0.01em !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    white-space: nowrap !important;
}

/* Tab hover effect */
.stTabs [data-baseweb="tab"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1) !important;
    border-color: var(--primary-blue-light) !important;
    background: var(--neutral-50) !important;
}

/* Active/Selected tab */
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    /* Light blue so tab label stays readable */
    background: linear-gradient(135deg, var(--primary-blue-200) 0%, var(--primary-blue-100) 100%) !important;
    color: var(--neutral-900) !important;
    border-color: var(--primary-blue-light) !important;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.18),
                0 2px 6px rgba(37, 99, 235, 0.10) !important;
    font-weight: 700 !important;
}

/* Tab panel (content area) */
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 2rem;
    animation: tabContentFadeIn 0.3s ease-out;
}

@keyframes tabContentFadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Hide inactive tab panels cleanly */
.stTabs [data-baseweb="tab-panel"][hidden] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}

/* ========================================================================
   MODERN CARDS & CONTAINERS
   ======================================================================== */

/* KPI Cards */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1.25rem;
    margin: 1.5rem 0 2rem 0;
    animation: slideInUp 0.5s ease-out;
}

@keyframes slideInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.kpi-card {
    background: white;
    border: 1px solid var(--neutral-200);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04),
                0 1px 3px rgba(15, 23, 42, 0.06);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary-blue), var(--primary-blue-light));
    opacity: 0;
    transition: opacity 0.3s ease;
}

.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08),
                0 4px 8px rgba(15, 23, 42, 0.04);
}

.kpi-card:hover::before {
    opacity: 1;
}

.kpi-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--neutral-500);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}

.kpi-value {
    font-size: 36px;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(135deg, currentColor 0%, currentColor 100%);
    -webkit-background-clip: text;
}

/* ========================================================================
   MODERN BUTTONS
   ======================================================================== */

/* Primary buttons */
.stButton > button[kind="primary"],
button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"],
button[data-testid="baseButton-primary"] {
    /* Light blue button so text remains visible */
    background: linear-gradient(135deg, var(--primary-blue-200) 0%, var(--primary-blue-100) 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: -0.01em !important;
    color: var(--neutral-900) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2),
                0 2px 4px rgba(37, 99, 235, 0.1) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stButton > button[kind="primary"]:hover,
button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    background: linear-gradient(135deg, #93c5fd 0%, var(--primary-blue-100) 100%) !important;
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22),
                0 4px 10px rgba(37, 99, 235, 0.12) !important;
}

.stButton > button[kind="primary"]:active,
button[kind="primary"]:active,
.stButton > button[data-testid="baseButton-primary"]:active,
button[data-testid="baseButton-primary"]:active {
    transform: translateY(0) !important;
    background: linear-gradient(135deg, var(--primary-blue-100) 0%, var(--primary-blue-50) 100%) !important;
}

/* Ensure all buttons have visible text */
/* IMPORTANT: Do NOT style all global `button` tags.
   Some Streamlit/BaseWeb widgets (Selectbox, DateInput, etc.) use button-like elements.
   Styling `button { color: ... }` will make selected values/placeholder text disappear. */
.stButton > button {
    color: var(--neutral-800) !important;
}

.stButton > button[kind="secondary"],
button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-secondary"] {
    color: var(--neutral-700) !important;
}


/* ========================================================================
   MODERN FORM INPUTS
   ======================================================================== */

/* Text inputs, text areas, number inputs */
input[type="text"],
input[type="number"],
input[type="email"],
textarea,
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    border-radius: 12px !important;
    border: 1.5px solid var(--neutral-300) !important;
    padding: 0.75rem 1rem !important;
    font-size: 15px !important;
    transition: all 0.3s ease !important;
    background: white !important;
    color: var(--neutral-900) !important;
    caret-color: var(--neutral-900) !important;
}

/* Placeholder text visibility */
input[type="text"]::placeholder,
input[type="number"]::placeholder,
input[type="email"]::placeholder,
textarea::placeholder,
.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
.stNumberInput input::placeholder {
    color: var(--neutral-500) !important;
    opacity: 1 !important;
}

input[type="text"]:focus,
input[type="number"]:focus,
input[type="email"]:focus,
textarea:focus,
.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {
    border-color: var(--primary-blue) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    outline: none !important;
}

/* Select boxes */
.stSelectbox > div > div {
    border-radius: 12px !important;
    border: 1.5px solid var(--neutral-300) !important;
    transition: all 0.3s ease !important;
    background: white !important;
}

.stSelectbox > div > div:hover {
    border-color: var(--primary-blue-light) !important;
}

/* Selected/focused "box" should show light-blue but keep text readable */
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within,
div[data-testid="stDateInput"] > div:focus-within,
.stSelectbox > div > div:focus-within,
.stDateInput > div > div:focus-within {
    border-color: var(--primary-blue-light) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
    background: var(--primary-blue-50) !important;
}

/* Selectbox / Multiselect text visibility (BaseWeb) */
div[data-testid="stSelectbox"] [data-baseweb="select"],
div[data-testid="stMultiSelect"] [data-baseweb="select"] {
    color: var(--neutral-900) !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] * ,
div[data-testid="stMultiSelect"] [data-baseweb="select"] * {
    color: var(--neutral-900) !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] input::placeholder,
div[data-testid="stMultiSelect"] [data-baseweb="select"] input::placeholder {
    color: var(--neutral-500) !important;
    opacity: 1 !important;
}

/* Date inputs */
.stDateInput > div > div {
    border-radius: 12px !important;
    border: 1.5px solid var(--neutral-300) !important;
    background: white !important;
}

div[data-testid="stDateInput"] input,
.stDateInput input {
    color: var(--neutral-900) !important;
    caret-color: var(--neutral-900) !important;
}

div[data-testid="stDateInput"] input::placeholder,
.stDateInput input::placeholder {
    color: var(--neutral-500) !important;
    opacity: 1 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--neutral-300) !important;
    border-radius: 16px !important;
    background: var(--neutral-50) !important;
    padding: 2rem !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--primary-blue) !important;
    background: white !important;
}

/* Checkboxes */
.stCheckbox {
    padding: 0.5rem 0 !important;
}

.stCheckbox > label {
    font-size: 15px !important;
    font-weight: 500 !important;
    color: var(--neutral-700) !important;
}

/* ========================================================================
   DATA TABLES
   ======================================================================== */

.stDataFrame {
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04),
                0 1px 3px rgba(15, 23, 42, 0.06) !important;
    animation: fadeIn 0.4s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.stDataFrame thead tr th {
    background: linear-gradient(135deg, var(--neutral-100) 0%, var(--neutral-50) 100%) !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    color: var(--neutral-800) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 1rem !important;
    border-bottom: 2px solid var(--neutral-300) !important;
}

.stDataFrame tbody tr {
    transition: all 0.2s ease !important;
    border-bottom: 1px solid var(--neutral-200) !important;
}

.stDataFrame tbody tr:hover {
    background: linear-gradient(90deg, rgba(37, 99, 235, 0.03) 0%, rgba(59, 130, 246, 0.02) 100%) !important;
    transform: scale(1.001) !important;
}

.stDataFrame tbody tr td {
    padding: 0.875rem 1rem !important;
    font-size: 14px !important;
    color: var(--neutral-700) !important;
}

/* ========================================================================
   PROGRESS BARS
   ======================================================================== */

/* NOTE:
   Streamlit renders progress using BaseWeb with role="progressbar".
   Styling wrapper divs can lead to a "double bar" artifact. Target the
   real progressbar element and its immediate fill only. */

/* Container spacing */
[data-testid="stProgress"], .stProgress {
    margin: 0.75rem 0 1rem 0 !important;
}

/* Hard-reset shadows/borders on progressbar subtree only */
[data-testid="stProgress"] [role="progressbar"],
[data-testid="stProgress"] [role="progressbar"] * {
    box-shadow: none !important;
    border: 0 !important;
    outline: none !important;
}

/* Track (single line) */
[data-testid="stProgress"] [role="progressbar"],
.stProgress [role="progressbar"] {
    background: var(--neutral-200) !important;
    border-radius: 999px !important;
    height: 12px !important;
    overflow: hidden !important;
    padding: 0 !important;
}

/* Fill */
[data-testid="stProgress"] [role="progressbar"] > div,
.stProgress [role="progressbar"] > div {
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    border-radius: 999px !important;
    background: linear-gradient(90deg, var(--primary-blue) 0%, var(--primary-blue-light) 100%) !important;
    animation: progressPulse 1.6s ease-in-out infinite;
}

@keyframes progressPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.9; }
}

/* ========================================================================
   ALERTS & MESSAGES
   ======================================================================== */

/* Success messages */
.stSuccess {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%) !important;
    border-left: 4px solid var(--success-green) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    color: #065f46 !important;
    font-weight: 500 !important;
    animation: slideInRight 0.4s ease-out;
}

@keyframes slideInRight {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

/* Error messages */
.stError {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%) !important;
    border-left: 4px solid var(--danger-red) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    color: #7f1d1d !important;
    font-weight: 500 !important;
    animation: shake 0.5s ease-in-out;
}

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-8px); }
    75% { transform: translateX(8px); }
}

/* Warning messages */
.stWarning {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%) !important;
    border-left: 4px solid var(--warning-orange) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    color: #78350f !important;
    font-weight: 500 !important;
}

/* Info messages */
.stInfo {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
    border-left: 4px solid var(--primary-blue) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    color: #1e3a8a !important;
    font-weight: 500 !important;
}

/* ========================================================================
   EXPANDERS & ACCORDIONS
   ======================================================================== */

.streamlit-expanderHeader {
    background: white !important;
    border: 1.5px solid var(--neutral-200) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    transition: all 0.3s ease !important;
}

.streamlit-expanderHeader:hover {
    background: var(--neutral-50) !important;
    border-color: var(--primary-blue-light) !important;
    transform: translateX(4px) !important;
}

.streamlit-expanderContent {
    border: 1.5px solid var(--neutral-200) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 1.5rem !important;
    background: white !important;
    animation: expandDown 0.3s ease-out;
}

@keyframes expandDown {
    from { opacity: 0; max-height: 0; }
    to { opacity: 1; max-height: 1000px; }
}

/* ========================================================================
   DIVIDERS
   ======================================================================== */

hr {
    margin: 2rem 0 !important;
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--neutral-300), transparent) !important;
}

/* ========================================================================
   SPINNER & LOADING STATES
   ======================================================================== */

.stSpinner > div {
    border-color: var(--primary-blue) !important;
    border-right-color: transparent !important;
    animation: spin 0.8s linear infinite !important;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* ========================================================================
   HEADERS & TYPOGRAPHY
   ======================================================================== */

h1, h2, h3, h4, h5, h6 {
    color: var(--neutral-900) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

h1 { font-size: 2.5rem !important; margin-bottom: 1.5rem !important; }
h2 { font-size: 2rem !important; margin-bottom: 1.25rem !important; }
h3 { font-size: 1.5rem !important; margin-bottom: 1rem !important; }

p, li, label {
    color: var(--neutral-700) !important;
    line-height: 1.6 !important;
}

/* ========================================================================
   DOWNLOAD BUTTONS & FORM SUBMIT BUTTONS
   ======================================================================== */

/* Download buttons */
.stDownloadButton > button,
button[data-testid="stDownloadButton"] {
    background: linear-gradient(135deg, var(--success-green) 0%, var(--success-green-dark) 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2) !important;
    transition: all 0.3s ease !important;
}

.stDownloadButton > button:hover,
button[data-testid="stDownloadButton"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3) !important;
}

/* Form submit buttons */
.stForm button[type="submit"],
button[type="submit"] {
    background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-blue-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    transition: all 0.3s ease !important;
}

.stForm button[type="submit"]:hover,
button[type="submit"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.28) !important;
}

/* ========================================================================
   CUSTOM SUCCESS CARDS (for post-submit views)
   ======================================================================== */

.success-card {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    padding: 2rem;
    border-radius: 16px;
    color: white;
    margin: 1.5rem 0;
    box-shadow: 0 12px 32px rgba(16, 185, 129, 0.24),
                0 4px 8px rgba(16, 185, 129, 0.12);
    animation: successPop 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

@keyframes successPop {
    0% { opacity: 0; transform: scale(0.8); }
    100% { opacity: 1; transform: scale(1); }
}

.success-card h3 {
    margin: 0 0 0.75rem 0;
    color: white !important;
    font-size: 1.5rem !important;
}

.success-card p {
    margin: 0.5rem 0;
    opacity: 0.95;
    color: white !important;
    font-size: 1rem !important;
}

/* ========================================================================
   FORM SECTIONS
   ======================================================================== */

.section-title {
    font-size: 1.125rem;
    font-weight: 700;
    color: var(--neutral-900);
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid var(--neutral-200);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.warning-banner {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-left: 4px solid var(--warning-orange);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    animation: slideInLeft 0.4s ease-out;
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(20px); }
    to { opacity: 1; transform: translateX(0); }
}

.warning-banner strong {
    color: #78350f;
}

/* ========================================================================
   RESPONSIVE DESIGN
   ======================================================================== */

@media (max-width: 768px) {
    .kpi-row {
        grid-template-columns: 1fr;
    }
    
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.75rem !important; }
}

/* ========================================================================
   PREVENT LAYOUT SHIFT & GLITCHES
   ======================================================================== */

/* Remove default Streamlit padding glitches */
.main > div:first-child {
    padding-top: 0 !important;
}

/* Smooth all transitions */
* {
    transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* Prevent text selection on buttons for cleaner UX */
button {
    user-select: none;
    -webkit-user-select: none;
}

/* Remove focus outline on click, keep for keyboard navigation */
button:focus:not(:focus-visible) {
    outline: none;
}

button:focus-visible {
    outline: 2px solid var(--primary-blue);
    outline-offset: 2px;
}

</style>
"""
