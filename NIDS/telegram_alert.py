#!/usr/bin/env python3
"""
Telegram Alert Module for Hybrid NIDPS using Telethon library
"""

import os
import asyncio
import datetime
from dotenv import load_dotenv
from telethon import TelegramClient

# Load environment variables
load_dotenv()

class TelegramAlerter:
    """Class for sending alerts via Telegram using Telethon."""
    
    def __init__(self, bot_token=None, chat_id=None, api_id=None, api_hash=None):
        """
        Initialize Telegram client with API credentials.
        
        Args:
            bot_token: Telegram bot token (defaults to env var)
            chat_id: Chat ID to send messages to (defaults to env var)
            api_id: Telegram API ID (defaults to env var)
            api_hash: Telegram API hash (defaults to env var)
        """
        # Load from parameters or environment variables
        self.bot_token = bot_token or os.getenv('TELEGRAM_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.api_id = api_id or os.getenv('TELEGRAM_API_ID')
        self.api_hash = api_hash or os.getenv('TELEGRAM_API_HASH')
        
        # Get the current script directory for session file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.session_path = os.path.join(current_dir, "nids_bot_session")
        
        # Initialize client
        self.client = None
        self.loop = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Telegram client once"""
        try:
            # Create new event loop
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Initialize client
            self.client = TelegramClient(
                self.session_path,
                self.api_id, 
                self.api_hash
            )
            
            # Start the client
            self.loop.run_until_complete(self.client.start(bot_token=self.bot_token))
            print("Telegram client initialized successfully")
        except Exception as e:
            print(f"Error initializing Telegram client: {e}")
            self.client = None
    
    def send_alert(self, message):
        """Send alert message via Telegram"""
        if not self.client:
            print("Telegram client not initialized")
            return False
            
        try:
            # Create async function to send message
            async def send_message():
                await self.client.send_message(
                    self.chat_id,
                    message,
                    parse_mode='html'
                )
            
            # Run the async function in the existing loop
            future = asyncio.run_coroutine_threadsafe(
                send_message(), 
                self.loop
            )
            # Wait for result with timeout
            future.result(timeout=30)
            return True
        except Exception as e:
            print(f"Error sending Telegram alert: {e}")
            return False
    
    def format_anomaly_alert(self, flow, anomaly_score, anomaly_details=None):
        """Format an anomaly alert message with HTML formatting."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"⚠️ <b>ANOMALY DETECTED</b> ⚠️\n\n"
        message += f"<b>Time:</b> {timestamp}\n"
        message += f"<b>Source IP:</b> {flow.get('saddr', 'Unknown')}\n"
        message += f"<b>Source Port:</b> {flow.get('sport', 'Unknown')}\n"
        message += f"<b>Destination IP:</b> {flow.get('daddr', 'Unknown')}\n"
        message += f"<b>Destination Port:</b> {flow.get('dport', 'Unknown')}\n"
        message += f"<b>Protocol:</b> {flow.get('proto', 'Unknown')}\n"
        
        # Add application protocol if available
        if 'appproto' in flow and flow['appproto']:
            message += f"<b>App Protocol:</b> {flow.get('appproto', 'Unknown')}\n"
            
        # Add flow timestamp if available
        if 'starttime' in flow and flow['starttime']:
            message += f"<b>Flow Start:</b> {flow.get('starttime', 'Unknown')}\n"
        
        message += f"<b>Anomaly Score:</b> {anomaly_score:.4f}\n"
        
        # Add anomaly details if available
        if anomaly_details and len(anomaly_details) > 0:
            message += "\n<b>Anomaly Details:</b>\n"
            for detail in anomaly_details[:3]:  # Limit to top 3 details
                feature = detail.get('feature', 'Unknown')
                value = detail.get('value', 0)
                z_score = detail.get('z_score', 0)
                message += f"- {feature}: {value} (z-score: {z_score:.2f})\n"
        
        # Add alert timestamp
        message += f"\n<i>Alert generated at {timestamp}</i>"
        
        return message

    def send_image_alert(self, flow, anomaly_score, image_path, caption=None):
        """Send an alert with an image via Telegram."""
        if not self.client:
            print("Telegram client not initialized")
            return False
            
        try:
            # Format the caption if not provided
            if caption is None:
                caption = self.format_anomaly_alert(flow, anomaly_score, [])
                
            # Truncate caption if too long (Telegram limit is 1024 characters)
            if len(caption) > 1000:
                caption = caption[:997] + "..."
            
            # Create async function to send file
            async def send_file():
                await self.client.send_file(
                    self.chat_id,
                    image_path,
                    caption=caption,
                    parse_mode='html'
                )
            
            # Run the async function in the existing loop
            future = asyncio.run_coroutine_threadsafe(
                send_file(), 
                self.loop
            )
            # Wait for result with timeout
            future.result(timeout=30)
            return True
        except Exception as e:
            print(f"Error sending Telegram image alert: {e}")
            return False
            
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.client and self.client.is_connected():
            # Create async function to disconnect
            async def disconnect():
                await self.client.disconnect()
                
            # Run the async function in the existing loop if it exists
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    disconnect(), 
                    self.loop
                )