import streamlit as st
# !! THE FIX IS HERE !!
# We import 'gotrue' as its own package, not from 'supabase'
from supabase import create_client, Client
import gotrue
# ---
from cryptography.fernet import Fernet
from typing import Dict, Optional, List, Tuple
import os

# --- Configuration ---
from utils import get_config, console

# Load Supabase config
SUPABASE_URL = get_config("SUPABASE_URL")
SUPABASE_KEY = get_config("SUPABASE_KEY")
# Load encryption key
FERNET_KEY = get_config("FERNET_KEY").encode() # Must be bytes

# Initialize clients
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    fernet = Fernet(FERNET_KEY)
except Exception as e:
    console.log(f"[bold red]FATAL: Could not initialize Supabase or Fernet. Check keys in config.yaml.[/bold red]")
    console.log(f"Error: {e}")
    st.error(f"Failed to initialize authentication backend: {e}")
    st.stop()
    

# --- NEW: Pure Supabase Auth Functions ---

def sign_up_user(email, password, full_name) -> Optional[gotrue.types.User]:
    """Signs up a new user in Supabase Auth."""
    try:
        # The 'user' attribute is on the 'Session' object, but sign_up returns UserResponse
        user_response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": { "data": { "full_name": full_name } }
        })
        st.success("Registration successful! Please check your email to verify.")
        return user_response.user
    except Exception as e:
        st.error(f"Error registering: {e}")
        return None

def sign_in_user(email, password) -> Optional[gotrue.types.User]:
    """Signs in an existing user using Supabase Auth."""
    try:
        session = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        st.success("Login successful!")
        return session.user
    except Exception as e:
        st.error(f"Error logging in: {e}")
        return None

def sign_out_user():
    """Signs out the current user."""
    supabase.auth.sign_out()
    st.success("Logged out successfully.")

def get_current_user() -> Optional[gotrue.types.User]:
    """Gets the currently authenticated user from the session."""
    try:
        # This will fail if no session is active
        user = supabase.auth.get_user().user
        return user
    except Exception:
        return None

# --- API Key Management (Securely in Supabase) ---

def encrypt_key(api_key: str) -> str:
    """Encrypts an API key using the Fernet key."""
    return fernet.encrypt(api_key.encode()).decode()

def decrypt_key(encrypted_key: str) -> str:
    """Decrypts an API key using the Fernet key."""
    return fernet.decrypt(encrypted_key.encode()).decode()

def get_user_api_keys(user_id: str) -> Optional[Dict[str, str]]:
    """
    Fetches and decrypts API keys for a given user_id from Supabase.
    """
    try:
        response = supabase.table("user_api_keys").select("*").eq("user_id", user_id).execute()
        if not response.data:
            return None # No keys found for this user
        
        encrypted_keys = response.data[0]
        decrypted_keys = {
            "google": decrypt_key(encrypted_keys['google_api_key']) if encrypted_keys.get('google_api_key') else None,
            "cohere": decrypt_key(encrypted_keys['cohere_api_key']) if encrypted_keys.get('cohere_api_key') else None,
            "tavily": decrypt_key(encrypted_keys['tavily_api_key']) if encrypted_keys.get('tavily_api_key') else None,
            "owm": decrypt_key(encrypted_keys['owm_api_key']) if encrypted_keys.get('owm_api_key') else None,
        }
        return decrypted_keys
    except Exception as e:
        console.log(f"[red]Error fetching API keys for user {user_id}: {e}[/red]")
        st.error(f"Could not fetch your API keys: {e}")
        return None

def save_user_api_keys(user_id: str, keys_dict: Dict[str, str]):
    """
    Encrypts and saves (upserts) a user's API keys to Supabase.
    """
    try:
        encrypted_keys = {
            "user_id": user_id,
            "google_api_key": encrypt_key(keys_dict["google"]) if keys_dict.get("google") else None,
            "cohere_api_key": encrypt_key(keys_dict["cohere"]) if keys_dict.get("cohere") else None,
            "tavily_api_key": encrypt_key(keys_dict["tavily"]) if keys_dict.get("tavily") else None,
            "owm_api_key": encrypt_key(keys_dict["owm"]) if keys_dict.get("owm") else None,
            "updated_at": "now()"
        }
        
        # Upsert = Update if exists, Insert if not
        supabase.table("user_api_keys").upsert(encrypted_keys, on_conflict="user_id").execute()
        st.success("API keys saved successfully!")
    except Exception as e:
        console.log(f"[red]Error saving API keys for user {user_id}: {e}[/red]")
        st.error(f"Could not save your API keys: {e}")

# --- Chat History Management (in Supabase) ---

def get_session_list(user_id: str) -> List[str]:
    """
    Fetches the list of unique, most recent session_ids for a user.
    """
    try:
        # We need to create a SQL function in Supabase for this to work
        # Go to Supabase > SQL Editor > New Query
        # Run:
        # CREATE OR REPLACE FUNCTION get_user_sessions(p_user_id UUID)
        # RETURNS TABLE(session_id TEXT, last_message_time TIMESTAMPTZ) AS $$
        # BEGIN
        #     RETURN QUERY
        #     SELECT
        #         s.session_id,
        #         MAX(s.created_at) AS last_message_time
        #     FROM
        #         chat_history s
        #     WHERE
        #         s.user_id = p_user_id
        #     GROUP BY
        #         s.session_id
        #     ORDER BY
        #         last_message_time DESC;
        # END;
        # $$ LANGUAGE plpgsql;

        response = supabase.rpc('get_user_sessions', {"p_user_id": user_id}).execute()
        if response.data:
            return [row['session_id'] for row in response.data]
        return []
    except Exception as e:
        console.log(f"[red]Error fetching session list for user {user_id}: {e}[/red]")
        st.error(f"Could not load chat sessions: {e}. Did you run the SQL function in Supabase?")
        return []

def get_chat_history(user_id: str, session_id: str) -> List[Tuple[str, str]]:
    """
    Fetches a specific chat history session from Supabase.
    Returns a list of tuples: [('HumanMessage', 'query'), ('AIMessage', 'response')]
    """
    try:
        response = supabase.table("chat_history").select("*") \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .order("created_at", desc=False) \
            .execute()
        
        history = []
        for message in response.data:
            history.append((message['message_type'], message['content']))
        return history
    except Exception as e:
        console.log(f"[red]Error fetching chat history for session {session_id}: {e}[/red]")
        st.error(f"Could not load chat history: {e}")
        return []

def save_chat_message(user_id: str, session_id: str, message_type: str, content: str):
    """
    Saves a single chat message to the Supabase database.
    """
    try:
        supabase.table("chat_history").insert({
            "user_id": user_id,
            "session_id": session_id,
            "message_type": message_type,
            "content": content
        }).execute()
    except Exception as e:
        console.log(f"[red]Error saving chat message for session {session_id}: {e}[/red]")
        # Don't show a UI error for this, just log it