#!/usr/bin/env python3
"""
Telegram Alert Module with Chat ID Testing
Tests different formats for chat IDs to find one that works
"""

import os
import asyncio
import logging
import datetime
import threading
import time
from threading import Event
from dotenv import load_dotenv
from telethon import TelegramClient, utils

# Setup logging
logger = logging.getLogger('hybrid-nids')

# Load environment variables
load_dotenv()

class TelegramAlerter:
    """Class for sending alerts via Telegram with chat ID testing"""
    
    def __init__(self, bot_token=None, chat_id=None, api_id=None, api_hash=None):
        """Initialize Telegram alerter with chat ID testing"""
        # Load from parameters or environment variables
        self.bot_token = bot_token or os.getenv('TELEGRAM_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.api_id = api_id or os.getenv('API_ID')
        self.api_hash = api_hash or os.getenv('API_HASH')
        
        # Store original chat ID to try different formats
        self.original_chat_id = self.chat_id
        self.chat_formats_to_try = []
        
        # Process chat_id - ensure it's in the correct format
        self.processed_chat_id = self._process_chat_id(self.chat_id)
        
        # Check if credentials are available
        if not all([self.bot_token, self.chat_id, self.api_id, self.api_hash]):
            missing = []
            if not self.bot_token: missing.append("TELEGRAM_TOKEN")
            if not self.chat_id: missing.append("TELEGRAM_CHAT_ID")
            if not self.api_id: missing.append("API_ID")
            if not self.api_hash: missing.append("API_HASH")
            
            logger.warning(f"Telegram credentials missing: {', '.join(missing)}. "
                          f"Set these in your .env file.")
            return
        
        # Get the current directory for session file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.session_path = os.path.join(current_dir, "nids_bot_session")
        
        # Connection state tracking
        self.connected = Event()
        self.client = None
        self.loop = None
        self.client_thread = None
        
        # Chat ID tracking
        self.valid_chat_id = None
        self.chat_id_tested = False
        
        # Message queue for reliability
        self.message_queue = []
        
        # Initialize and connect
        logger.info("Initializing Telegram connection...")
        self._start_client_in_thread()
        
        # Wait for initial connection (but don't block too long)
        if not self.connected.wait(timeout=15):
            logger.warning("Initial Telegram connection not established within timeout")
            logger.warning("Messages will be queued and sent when connection is established")
    
    def _process_chat_id(self, chat_id):
        """Process chat_id to ensure it's in a format that the bot can use"""
        if not chat_id:
            return None
            
        # Store formats to try
        self.chat_formats_to_try = []
            
        # Try to convert to integer (preferred format for bots)
        try:
            int_id = int(chat_id)
            self.chat_formats_to_try.append(int_id)
            
            # For group chats, also try with additional '-100' prefix if it's negative
            if int_id < 0:
                # Convert to canonical supergroup format if it's not already
                if not str(abs(int_id)).startswith('100'):
                    str_val = str(abs(int_id))
                    supergroup_id = -int(f"100{str_val}")
                    self.chat_formats_to_try.append(supergroup_id)
            
            return int_id
        except ValueError:
            # Not an integer, check if it's a username
            if chat_id.startswith('@'):
                username = chat_id
                self.chat_formats_to_try.append(username)
                # Also try without @
                self.chat_formats_to_try.append(chat_id[1:])
                return username
            
            # If it's not an integer or username with @, prefix it with @
            username_with_at = f"@{chat_id}"
            self.chat_formats_to_try.append(username_with_at)
            self.chat_formats_to_try.append(chat_id)
            return username_with_at
    
    def _start_client_in_thread(self):
        """Start the client in a background thread"""
        try:
            # Create a new loop for the client
            self.loop = asyncio.new_event_loop()
            
            # Start client in a thread
            self.client_thread = threading.Thread(
                target=self._run_client_thread,
                daemon=True
            )
            self.client_thread.start()
            logger.debug("Client thread started")
        except Exception as e:
            logger.error(f"Error starting client thread: {e}")
    
    def _run_client_thread(self):
        """Thread function to run the client's event loop"""
        try:
            # Set this thread's event loop
            asyncio.set_event_loop(self.loop)
            
            # Run initialization
            self.loop.run_until_complete(self._init_client())
            
            # Run event loop to keep client alive
            self.loop.run_forever()
        except Exception as e:
            logger.error(f"Error in client thread: {e}")
            self.connected.clear()
    
    async def _init_client(self):
        """Initialize and connect the client"""
        try:
            # Convert API_ID to int
            api_id_int = int(self.api_id) if self.api_id else None
            
            # Create client
            self.client = TelegramClient(
                self.session_path,
                api_id_int,
                self.api_hash,
                loop=self.loop
            )
            
            # Connect to Telegram
            logger.info("Connecting to Telegram...")
            await self.client.connect()
            
            # Check if we're connected
            if not await self.client.is_user_authorized():
                # Need to login with bot token
                logger.info("Logging in with bot token...")
                await self.client.start(bot_token=self.bot_token)
            
            # Get bot info
            me = await self.client.get_me()
            logger.info(f"Connected to Telegram as @{me.username}")
            
            # Test the chat ID
            await self._test_chat_id_formats()
            
            # Set connected flag
            self.connected.set()
            
            # Process any queued messages
            if self.message_queue:
                logger.info(f"Processing {len(self.message_queue)} queued messages")
                for message in self.message_queue:
                    await self._send_message_internal(message)
                self.message_queue.clear()
            
            return True
        except Exception as e:
            logger.error(f"Error initializing Telegram client: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.connected.clear()
            return False
    
    async def _test_chat_id_formats(self):
        """Test different chat ID formats to find one that works"""
        if self.chat_id_tested:
            return
            
        logger.info("Testing different chat ID formats...")
        
        # First, try to get a list of dialogs (chats where the bot is a member)
        try:
            dialogs = await self.client.get_dialogs()
            logger.info(f"Bot is a member of {len(dialogs)} chats")
            
            # Log the details of each dialog for debugging
            for i, dialog in enumerate(dialogs):
                chat_id = utils.get_peer_id(dialog.entity)
                chat_name = dialog.name
                logger.info(f"Dialog {i+1}: {chat_name} (ID: {chat_id})")
                
                # Check if this matches our target chat ID
                if str(chat_id) == str(self.processed_chat_id) or str(chat_id) == str(self.original_chat_id):
                    logger.info(f"Found matching chat: {chat_name} (ID: {chat_id})")
                    self.valid_chat_id = chat_id
                    self.chat_id_tested = True
                    return
        except Exception as e:
            logger.warning(f"Failed to get dialogs: {e}")
        
        # If we didn't find a match, try all the formats
        for i, chat_format in enumerate(self.chat_formats_to_try):
            try:
                logger.info(f"Testing chat ID format {i+1}: {chat_format}")
                entity = await self.client.get_entity(chat_format)
                chat_id = utils.get_peer_id(entity)
                logger.info(f"Successfully resolved chat ID format: {chat_format} -> {chat_id}")
                self.valid_chat_id = chat_format
                self.chat_id_tested = True
                return
            except Exception as e:
                logger.warning(f"Chat ID format {chat_format} failed: {e}")
                continue
        
        # If we get here, none of the formats worked
        logger.error("Failed to find a valid chat ID format")
        logger.error("Make sure the bot is a member of the chat/group/channel")
        logger.error("For group chats, add the bot to the group first")
        
        # Suggest creating a new group
        logger.info("TIP: Try creating a new group in Telegram, add your bot to it, then get the chat ID")
        
        self.chat_id_tested = True
    
    def send_message(self, message):
        """Send a message with robust connection handling"""
        # Queue message if client not connected
        if not self.connected.is_set():
            logger.warning("Client not connected, queuing message")
            self.message_queue.append(message)
            return False
        
        # Send message if connected
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._send_message_internal(message),
                self.loop
            )
            return future.result(timeout=30)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            # Queue message on error
            self.message_queue.append(message)
            return False
    
    async def _ensure_connected(self):
        """Ensure client is connected, reconnect if needed"""
        if not self.client:
            logger.error("Client not initialized")
            return False
            
        if not self.client.is_connected():
            logger.info("Client disconnected, reconnecting...")
            try:
                await self.client.connect()
                if not await self.client.is_user_authorized():
                    await self.client.start(bot_token=self.bot_token)
                logger.info("Reconnected successfully")
                self.connected.set()
                return True
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                self.connected.clear()
                return False
        
        return True
    
    async def _send_message_internal(self, message):
        """Internal method to send a message"""
        # Double-check connection
        if not await self._ensure_connected():
            logger.error("Cannot send message: Not connected")
            self.message_queue.append(message)
            return False
            
        # If we haven't found a valid chat ID yet, test them
        if not self.chat_id_tested:
            await self._test_chat_id_formats()
        
        # Use the valid chat ID if we found one
        chat_id_to_use = self.valid_chat_id if self.valid_chat_id else self.processed_chat_id
        
        try:
            # Send message
            await self.client.send_message(chat_id_to_use, message)
            logger.info(f"Message sent successfully to {chat_id_to_use}")
            return True
        except Exception as e:
            logger.error(f"Error sending message to {chat_id_to_use}: {e}")
            
            # If we haven't tried all formats yet
            if self.valid_chat_id is None and len(self.chat_formats_to_try) > 0:
                # Try the other formats
                for chat_format in self.chat_formats_to_try:
                    if chat_format == chat_id_to_use:
                        continue
                    
                    try:
                        logger.info(f"Trying alternative chat ID: {chat_format}")
                        await self.client.send_message(chat_format, message)
                        logger.info(f"Message sent successfully to {chat_format}")
                        self.valid_chat_id = chat_format
                        return True
                    except Exception as inner_e:
                        logger.warning(f"Alternative format {chat_format} failed: {inner_e}")
            
            logger.error("All chat ID formats failed")
            return False
    
# Update the format_anomaly_alert method in TelegramAlerter class

def format_anomaly_alert(self, alert_data):
    """Format an anomaly alert message with emphasis on ML detection"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session = alert_data.get('session', {})
    
    # Create a more compact, ML-focused alert format
    message = f"🚨 ATTACK DETECTED 🚨\n\n"
    
    # ML Detection Scores - Show this prominently at the top
    ml_result = alert_data.get('ml_result', {})
    stat_result = alert_data.get('stat_result', {})
    
    message += "📊 ML DETECTION SCORES:\n"
    message += f"➡️ Decision Tree: {ml_result.get('dt_confidence', 0):.2f}\n"
    message += f"➡️ Random Forest: {ml_result.get('rf_confidence', 0):.2f}\n"
    
    # Add XGBoost if available
    if 'xgb_confidence' in ml_result:
        message += f"➡️ XGBoost: {ml_result.get('xgb_confidence', 0):.2f}\n"
        
    # Overall score
    message += f"⭐ COMBINED SCORE: {alert_data.get('combined_score', 0):.2f}\n\n"
    
    # Connection details - make this more compact
    message += "🔗 CONNECTION DETAILS:\n"
    
    # Check if this is a critical service and highlight it
    is_critical = False
    try:
        port = int(alert_data.get('dst_port', 0))
        if port in [22, 23, 21, 3389, 445, 139, 1433, 3306]:
            is_critical = True
            message += f"⚠️ CRITICAL SERVICE: Port {port}\n"
    except (ValueError, TypeError):
        pass
        
    message += f"Source: {alert_data.get('src_ip', 'Unknown')}:{alert_data.get('src_port', 'Unknown')}\n"
    message += f"Destination: {alert_data.get('dst_ip', 'Unknown')}:{alert_data.get('dst_port', 'Unknown')}\n"
    message += f"Protocol: {alert_data.get('proto', 'Unknown')}/{alert_data.get('app_proto', 'Unknown')}\n"
    
    # Detection timing information
    if 'detection_latency' in alert_data:
        message += f"🕒 Detection Latency: {alert_data.get('detection_latency', 0):.2f}s\n"
    
    # Add incremental detection indicator if applicable
    if alert_data.get('is_incremental', False):
        message += "⚡ Real-time Detection (active flow)\n"
    
    # Traffic statistics in a compact format
    message += f"\n📊 TRAFFIC:\n"
    message += f"Bytes: {alert_data.get('total_bytes', 0):,} | "
    message += f"Packets: {alert_data.get('total_packets', 0):,} | "
    message += f"Duration: {alert_data.get('duration', 0):.2f}s\n"
    
    # Add attack type indicators based on port/protocol
    message += "\n🔍 ATTACK INDICATORS:\n"
    
    port = alert_data.get('dst_port', '')
    try:
        port_num = int(port)
        
        if port_num == 22:
            message += "• SSH brute force attempt detected\n"
        elif port_num == 23:
            message += "• Telnet intrusion detected\n"
        elif port_num == 3389:
            message += "• RDP attack detected\n"
        elif port_num == 21:
            message += "• FTP attack detected\n"
        elif port_num in [445, 139]:
            message += "• SMB/NetBIOS attack detected\n"
        elif port_num in [1433, 3306]:
            message += "• Database attack detected\n"
        elif port_num == 80 or port_num == 443:
            message += "• Web attack detected\n"
    except (ValueError, TypeError):
        pass
    
    # Check application layer details
    app_layer_anomalies = []
    if session.get('http_event_count', 0) > 0:
        if session.get('http_error', False):
            app_layer_anomalies.append("HTTP errors detected")
    
    if session.get('dns_event_count', 0) > 0:
        if session.get('dns_failure', False):
            app_layer_anomalies.append("DNS failures detected")
            
    if app_layer_anomalies:
        for anomaly in app_layer_anomalies:
            message += f"• {anomaly}\n"
    
    # Add timestamp
    message += f"\n⏰ Alert Time: {timestamp}\n"
    
    return message