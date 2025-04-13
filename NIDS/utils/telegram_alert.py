#!/usr/bin/env python3
"""
Telegram Alert Module for Hybrid NIDS using Telethon
"""

import os
import asyncio
import logging
import datetime
from dotenv import load_dotenv
from telethon import TelegramClient

# Setup logging
logger = logging.getLogger('hybrid-nids')

# Load environment variables
load_dotenv()

class TelegramAlerter:
    """Class for sending alerts via Telegram using Telethon"""
    
    def __init__(self, bot_token=None, chat_id=None, api_id=None, api_hash=None):
        """Initialize Telegram client"""
        # Load from parameters or environment variables
        self.bot_token = bot_token or os.getenv('TELEGRAM_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.api_id = api_id or os.getenv('API_ID')
        self.api_hash = api_hash or os.getenv('API_HASH')
        
        # Check if credentials are available
        if not all([self.bot_token, self.chat_id, self.api_id, self.api_hash]):
            missing = []
            if not self.bot_token: missing.append("TELEGRAM_TOKEN")
            if not self.chat_id: missing.append("TELEGRAM_CHAT_ID")
            if not self.api_id: missing.append("API_ID")
            if not self.api_hash: missing.append("API_HASH")
            
            logger.warning(f"Telegram credentials missing: {', '.join(missing)}. "
                          f"Set these in your .env file.")
            self.client = None
            return
        
        # Get the current directory for session file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.session_path = os.path.join(current_dir, "nids_bot_session")
        
        # Initialize client
        self.client = None
        self.loop = None
        
        # Initialize the client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Telegram client"""
        try:
            # Create event loop
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # Initialize client
            self.client = TelegramClient(
                self.session_path,
                self.api_id,
                self.api_hash
            ).start(bot_token=self.bot_token)

           
            

            # Start client
            with self.client:
                self.client.run_until_disconnected()
            logger.info("Telegram alerter is ready")
            
        except Exception as e:
            logger.error(f"Error initializing Telegram client: {e}")
            self.client = None
    
    def send_alert(self, message):
        """Send alert message via Telegram"""
        if not self.client:
            logger.error("Telegram client not initialized")
            return False

        try:
            async def send_message():
                try:
                    # Replace <br> with newlines for Telethon
                    message_text = message.replace('<br>', '\n')
                    
                    # Send the message
                    await self.client.send_message(
                        self.chat_id,
                        message_text
                    )
                    return True
                except Exception as e:
                    logger.error(f"Error sending Telegram message: {e}")
                    return False

            future = asyncio.run_coroutine_threadsafe(send_message(), self.loop)
            result = future.result(timeout=30)
            
            if result:
                logger.info("Telegram alert sent successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
            return False
    
    def format_anomaly_alert(self, flow, anomaly_score, anomaly_details=None):
        """Format an anomaly alert message"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"⚠️ ANOMALY DETECTED ⚠️\n\n"
        message += f"Time: {timestamp}\n"
        message += f"Source IP: {flow.get('saddr', 'Unknown')}\n"
        message += f"Source Port: {flow.get('sport', 'Unknown')}\n"
        message += f"Destination IP: {flow.get('daddr', 'Unknown')}\n"
        message += f"Destination Port: {flow.get('dport', 'Unknown')}\n"
        message += f"Protocol: {flow.get('proto', 'Unknown')}\n"
        
        # Add application protocol if available
        if 'appproto' in flow and flow['appproto']:
            message += f"App Protocol: {flow.get('appproto', 'Unknown')}\n"
            
        # Add flow timestamp if available
        if 'starttime' in flow and flow['starttime']:
            message += f"Flow Start: {flow.get('starttime', 'Unknown')}\n"
        
        message += f"Anomaly Score: {anomaly_score:.4f}\n"
        
        # Add anomaly details if available
        if anomaly_details and len(anomaly_details) > 0:
            message += "\nAnomaly Details:\n"
            for detail in anomaly_details[:3]:  # Limit to top 3 details
                feature = detail.get('feature', 'Unknown')
                value = detail.get('value', 0)
                z_score = detail.get('z_score', 0)
                message += f"- {feature}: {value} (z-score: {z_score:.2f})\n"
        
        # Add alert timestamp
        message += f"\nAlert generated at {timestamp}"
        
        return message
    
    