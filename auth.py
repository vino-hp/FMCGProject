"""
Authentication module for FMCG Demand Forecasting App
Handles user registration, login, session management, and protected routes.
Credentials are stored in a local JSON file (users.json).
"""
import json
import hashlib
import os
import re
import streamlit as st
from typing import Tuple, Optional

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")


# ──────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────

def _load_users() -> dict:
    """Load users dict from JSON file. Returns empty dict if file missing."""
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: dict) -> None:
    """Persist users dict to JSON file."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str) -> str:
    """Return a SHA-256 hex digest of the password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────

def _validate_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email.strip()))


def _validate_password(password: str) -> Tuple[bool, str]:
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    return True, ""


def _validate_name(name: str) -> Tuple[bool, str]:
    if len(name.strip()) < 2:
        return False, "Name must be at least 2 characters."
    return True, ""


# ──────────────────────────────────────────────
# Core auth functions
# ──────────────────────────────────────────────

def register_user(name: str, email: str, password: str) -> Tuple[bool, str]:
    """
    Register a new user.
    Returns (success: bool, message: str).
    """
    name = name.strip()
    email = email.strip().lower()
    password = password.strip()

    # Validate name
    ok, msg = _validate_name(name)
    if not ok:
        return False, msg

    # Validate email
    if not _validate_email(email):
        return False, "Please enter a valid email address."

    # Validate password
    ok, msg = _validate_password(password)
    if not ok:
        return False, msg

    users = _load_users()

    if email in users:
        return False, "An account with this email already exists."

    users[email] = {
        "name": name,
        "email": email,
        "password_hash": _hash_password(password),
    }
    _save_users(users)
    return True, f"Account created successfully! Welcome, {name}."


def login_user(email: str, password: str) -> Tuple[bool, str, Optional[dict]]:
    """
    Verify credentials.
    Returns (success: bool, message: str, user_info: dict | None).
    """
    email = email.strip().lower()
    password = password.strip()

    if not email or not password:
        return False, "Please enter both email and password.", None

    if not _validate_email(email):
        return False, "Please enter a valid email address.", None

    users = _load_users()

    user = users.get(email)
    if user is None:
        return False, "No account found with that email.", None

    if user["password_hash"] != _hash_password(password):
        return False, "Incorrect password. Please try again.", None

    return True, f"Welcome back, {user['name']}!", user


# ──────────────────────────────────────────────
# Session state helpers
# ──────────────────────────────────────────────

def init_auth_state() -> None:
    """Initialise auth-related session state keys."""
    auth_defaults = {
        "authenticated": False,
        "user_info": None,
        "auth_page": "login",  # "login" | "register"
    }
    for key, val in auth_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def is_authenticated() -> bool:
    return st.session_state.get("authenticated", False)


def logout() -> None:
    """Clear auth state and reset to login."""
    st.session_state["authenticated"] = False
    st.session_state["user_info"] = None
    st.session_state["auth_page"] = "login"


# ──────────────────────────────────────────────
# UI – Login page
# ──────────────────────────────────────────────

def show_login_page() -> None:
    st.markdown(
        """
        <style>
        .auth-container{max-width:420px;margin:3rem auto 0 auto;}
        .auth-title{font-size:1.8rem;color:#1f77b4;text-align:center;margin-bottom:.3rem;}
        .auth-subtitle{text-align:center;color:#666;margin-bottom:1.5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">📈 FMCG Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">Sign in to your account</div>', unsafe_allow_html=True)

    with st.form("login_form"):
        email = st.text_input("📧 Email", placeholder="you@example.com")
        password = st.text_input("🔒 Password", type="password", placeholder="Your password")
        submitted = st.form_submit_button("Login", use_container_width=True, type="primary")

    if submitted:
        success, message, user_info = login_user(email, password)
        if success:
            st.session_state["authenticated"] = True
            st.session_state["user_info"] = user_info
            st.success(message)
            st.rerun()
        else:
            st.error(message)

    st.markdown("---")
    st.markdown("Don't have an account?")
    if st.button("Create an account →", use_container_width=True):
        st.session_state["auth_page"] = "register"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# UI – Register page
# ──────────────────────────────────────────────

def show_register_page() -> None:
    st.markdown(
        """
        <style>
        .auth-container{max-width:420px;margin:3rem auto 0 auto;}
        .auth-title{font-size:1.8rem;color:#1f77b4;text-align:center;margin-bottom:.3rem;}
        .auth-subtitle{text-align:center;color:#666;margin-bottom:1.5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">📈 FMCG Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">Create a new account</div>', unsafe_allow_html=True)

    with st.form("register_form"):
        name = st.text_input("👤 Full Name", placeholder="Jane Doe")
        email = st.text_input("📧 Email", placeholder="you@example.com")
        password = st.text_input("🔒 Password", type="password", placeholder="At least 6 characters")
        confirm = st.text_input("🔒 Confirm Password", type="password", placeholder="Repeat your password")
        submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")

    if submitted:
        if password != confirm:
            st.error("Passwords do not match.")
        else:
            success, message = register_user(name, email, password)
            if success:
                st.success(message)
                st.info("You can now log in with your new credentials.")
                st.session_state["auth_page"] = "login"
                st.rerun()
            else:
                st.error(message)

    st.markdown("---")
    st.markdown("Already have an account?")
    if st.button("← Back to Login", use_container_width=True):
        st.session_state["auth_page"] = "login"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Gate: show auth pages or allow app through
# ──────────────────────────────────────────────

def require_auth() -> bool:
    """
    Call at the top of main(). Returns True if the user is authenticated
    (app should proceed), False if an auth page was rendered (app should stop).
    """
    init_auth_state()

    if is_authenticated():
        return True

    # Not authenticated — show login or register
    if st.session_state["auth_page"] == "register":
        show_register_page()
    else:
        show_login_page()

    return False
