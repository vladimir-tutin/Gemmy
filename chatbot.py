import discord
from discord.ext import commands, tasks
import google.generativeai as genai
import aiohttp
import traceback
import logging
import asyncio
import os
import shelve
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import json
import re
import trafilatura
import markdownify
import random
from datetime import datetime
import requests  # For Pastee upload (and fallback Pastebin)
from urllib.parse import urlparse # Import for URL parsing
import http

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get environment variables
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY")
GUILD_ID_STR = os.getenv("GUILD_ID", "0")  # Get as string initially
PASTEBIN_API_KEY = os.getenv("PASTEBIN_API_KEY")  # Add Pastebin API key
PASTEE_BEARER_TOKEN = os.getenv("PASTEE_BEARER_TOKEN") # Add Pastee API key

try:
    GUILD_ID = int(GUILD_ID_STR)
except ValueError:
    logger.error(f"Invalid GUILD_ID: {GUILD_ID_STR}.  Must be an integer.")
    GUILD_ID = 0

if not DISCORD_BOT_TOKEN or not GOOGLE_AI_KEY or not GUILD_ID or not PASTEBIN_API_KEY or not PASTEE_BEARER_TOKEN: # PASTEE_BEARER_TOKEN is now required
    logger.error("Missing required environment variables (DISCORD_BOT_TOKEN, GOOGLE_AI_KEY, GUILD_ID, PASTEBIN_API_KEY, PASTEE_BEARER_TOKEN). Please check your .env file.") # PASTEE_BEARER_TOKEN is now required
    exit()

# Define intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True  # For member-related features

# Initialize bot
bot = commands.Bot(command_prefix="!", intents=intents)

# Global variables for tracking settings and conversation history
smart_replies_status: Dict[int, bool] = {}
tracked_channels: List[int] = []
tracked_threads: List[int] = []
message_history: Dict[int, genai.ChatSession] = {}
last_bot_question: Dict[int, str] = {}

# Data Persistence
DATA_FILE = "chatdata"               # Shelve data file
MODIFIER_FILE = "relevance_modifiers.txt"  # File for relevance modifiers
PERSONALITY_FILE = "channel_personalities.json"     # File for bot personalities

# Gemini AI Configuration and prompt template
model = None  # type: Optional[genai.GenerativeModel]
text_generation_config = None
safety_settings = None
bot_template: List[Dict[str, List[str]]] = []  # To be built in configure_gemini()

# Recursion limit for AI response augmentation
MAX_AI_RESPONSE_DEPTH = 3

# Global aiohttp session and a simple search cache
http_session: Optional[aiohttp.ClientSession] = None
searx_cache: Dict[str, List[str]] = {}

# Gemini AI and personality instructions
bot_instructions = (
    "Alright, I'm Gemmy. Prepare for some sass along with your answers. "
    "If a question requires current information (news, weather, etc.), you MUST use “NEED_WEB_SEARCH {query}” to find the answer. "
    "I'll give you the straight-up URL for links. Web page content? Bring it on. "
    "If you ask about previous discord conversations I'll respond with “READ_HISTORY”."
)

def load_personalities_sync() -> Dict[int, str]:
    """Loads channel-specific personalities from a JSON file."""
    try:
        with open(PERSONALITY_FILE, "r", encoding="utf-8") as f:
            personalities = json.load(f)
        logger.info("Channel personalities loaded successfully.")
        return {int(k): v for k, v in personalities.items()}  # Ensure keys are integers
    except FileNotFoundError:
        logger.warning("Channel personalities file not found. Creating a default one.")
        return {}
    except json.JSONDecodeError:
        logger.error("Error decoding channel personalities JSON. Returning an empty dictionary.")
        return {}
    except Exception as e:
        logger.error(f"Error loading channel personalities: {e}")
        return {}

def save_personalities_sync(personalities: Dict[int, str]):
    """Saves channel-specific personalities to a JSON file."""
    try:
        with open(PERSONALITY_FILE, "w", encoding="utf-8") as f:
            json.dump(personalities, f, indent=4)
        logger.info("Channel personalities saved successfully.")
    except Exception as e:
        logger.error(f"Error saving channel personalities: {e}")

async def load_personalities() -> Dict[int, str]:
    """Loads channel-specific personalities asynchronously."""
    return await asyncio.to_thread(load_personalities_sync)

async def save_personalities(personalities: Dict[int, str]):
    """Saves channel-specific personalities asynchronously."""
    await asyncio.to_thread(save_personalities_sync, personalities)

def get_personality_for_channel(channel_id: int, personalities: Dict[int, str]) -> str:
    """Gets the personality for a specific channel."""
    return personalities.get(channel_id, "You are a helpful discord bot.")  # Default personality

async def configure_gemini(channel_id: int):
    """Configures the Gemini AI model and the bot template using the channel-specific personality."""
    global model, text_generation_config, safety_settings, bot_template
    genai.configure(api_key=GOOGLE_AI_KEY)
    text_generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    personalities = load_personalities_sync()  # Load personalities synchronously
    personality = get_personality_for_channel(channel_id, personalities)
    full_prompt = personality + "\n" + bot_instructions
    # Build the initial prompt template (a list of message dictionaries)
    bot_template = [
        {"role": "user", "parts": [full_prompt]},
        {"role": "model", "parts": ["Understood."]},
    ]
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=text_generation_config,
        safety_settings=safety_settings
    )
    logger.info(f"Gemini model configured for channel {channel_id}.")

def get_history_key(channel_id: int) -> str:
    """Generates a key for storing channel history in shelve."""
    return f"history_{channel_id}"

async def load_data():
    """Loads persisted data from shelve."""
    global tracked_threads, smart_replies_status, message_history, tracked_channels
    try:
        with shelve.open(DATA_FILE) as file:
            tracked_threads = file.get('tracked_threads', [])
            smart_replies_status = file.get('smart_replies_status', {})
            tracked_channels = file.get('tracked_channels', [])
            for key in file:
                if key.isnumeric():
                    channel_id = int(key)
                    history = file.get(get_history_key(channel_id), [])
                    if smart_replies_status.get(channel_id, False):
                        adjusted_history = bot_template.copy()
                        adjusted_history.extend([
                            {'role': 'user', 'parts': ["You are in a channel with smart replies enabled. Respond to every message here, even if not mentioned."]},
                            {'role': 'model', 'parts': ["Understood! I'll respond to all messages in this channel."]},
                        ])
                        adjusted_history.extend(history[2:])
                        message_history[channel_id] = model.start_chat(history=adjusted_history)
                    else:
                        message_history[channel_id] = model.start_chat(history=bot_template)
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")

async def persist_data():
    """Persists settings and conversation history to shelve."""
    try:
        with shelve.open(DATA_FILE) as file:
            file['tracked_threads'] = tracked_threads
            file['smart_replies_status'] = smart_replies_status
            file['tracked_channels'] = tracked_channels
            for channel_id, chat_session in message_history.items():
                file[get_history_key(channel_id)] = chat_session.history
        logger.info("Data persisted successfully.")
    except Exception as e:
        logger.error(f"Error persisting data: {e}")

async def load_modifiers() -> List[str]:
    """Loads relevance modifiers from file."""
    try:
        with open(MODIFIER_FILE, "r", encoding="utf-8") as f:
            modifiers = [line.strip() for line in f if line.strip()]
        logger.info("Relevance modifiers loaded successfully.")
        return modifiers
    except FileNotFoundError:
        logger.warning("Relevance modifiers file not found. Creating an empty one.")
        open(MODIFIER_FILE, "w", encoding="utf-8").close()
        return []
    except Exception as e:
        logger.error(f"Error loading relevance modifiers: {e}")
        return []

async def save_modifiers(modifiers: List[str]):
    """Saves relevance modifiers to file."""
    try:
        with open(MODIFIER_FILE, "w", encoding="utf-8") as f:
            for modifier in modifiers:
                f.write(modifier + "\n")
        logger.info("Relevance modifiers saved successfully.")
    except Exception as e:
        logger.error(f"Error saving relevance modifiers: {e}")

# ----------------------------
# Slash Commands
# ----------------------------

@bot.tree.command(name="enable_smart_replies", description="Enable smart replies in this channel.", guild=discord.Object(id=GUILD_ID))
async def enable_smart_replies(interaction: discord.Interaction):
    channel_id = interaction.channel.id
    smart_replies_status[channel_id] = True
    await configure_gemini(channel_id)
    temp_template = bot_template.copy()
    temp_template.extend([
        {'role': 'user', 'parts': ["You are in a channel with smart replies enabled. Respond to every message here, even if not mentioned."]},
        {'role': 'model', 'parts': ["Understood! I'll respond to all messages in this channel."]},
    ])
    message_history[channel_id] = model.start_chat(history=temp_template)
    await persist_data()
    await interaction.response.send_message("Smart replies enabled for this channel.", ephemeral=True)

@bot.tree.command(name="disable_smart_replies", description="Disable smart replies in this channel.", guild=discord.Object(id=GUILD_ID))
async def disable_smart_replies(interaction: discord.Interaction):
    channel_id = interaction.channel.id
    smart_replies_status[channel_id] = False
    await configure_gemini(channel_id)
    message_history[channel_id] = model.start_chat(history=bot_template)
    await persist_data()
    await interaction.response.send_message("Smart replies disabled for this channel.", ephemeral=True)

@bot.tree.command(name="track_channel", description="Make the bot always respond in this channel.", guild=discord.Object(id=GUILD_ID))
async def track_channel(interaction: discord.Interaction):
    channel_id = interaction.channel.id
    if channel_id not in tracked_channels:
        tracked_channels.append(channel_id)
        await persist_data()
        await interaction.response.send_message("This channel has been added to the tracked channels.", ephemeral=True)
    else:
        await interaction.response.send_message("This channel is already being tracked.", ephemeral=True)

@bot.tree.command(name="untrack_channel", description="Stop the bot from always responding in this channel.", guild=discord.Object(id=GUILD_ID))
async def untrack_channel(interaction: discord.Interaction):
    channel_id = interaction.channel.id
    if channel_id in tracked_channels:
        tracked_channels.remove(channel_id)
        await persist_data()
        await interaction.response.send_message("This channel has been removed from the tracked channels.", ephemeral=True)
    else:
        await interaction.response.send_message("This channel is not being tracked.", ephemeral=True)

@bot.tree.command(name="adjust_relevance", description="Adjust bot's relevance (include or exclude a keyword).", guild=discord.Object(id=GUILD_ID))
async def adjust_relevance(interaction: discord.Interaction, reason: str, exclude: bool = False):
    modifiers = await load_modifiers()
    modifier_text = f"{'EXCLUDE' if exclude else 'INCLUDE'}: {reason}"
    if modifier_text not in modifiers:
        modifiers.append(modifier_text)
        await save_modifiers(modifiers)
        await interaction.response.send_message(f"Added modifier: {modifier_text}", ephemeral=True)
    else:
        await interaction.response.send_message(f"Modifier already exists: {modifier_text}", ephemeral=True)

@bot.tree.command(name="set_personality", description="Set the bot's personality.", guild=discord.Object(id=GUILD_ID))
async def set_personality(interaction: discord.Interaction, personality: str):
    channel_id = interaction.channel.id
    personalities = await load_personalities()
    personalities[channel_id] = personality
    await save_personalities(personalities)
    await configure_gemini(channel_id)  # Reconfigure Gemini with new personality
    # Reinitialize chat histories with the new prompt
    for chan_id, chat_session in message_history.items():
        if smart_replies_status.get(chan_id, False):
            adjusted_history = bot_template.copy()
            adjusted_history.extend([
                {'role': 'user', 'parts': ["You are in a channel with smart replies enabled. Respond to every message here, even if not mentioned."]},
                {'role': 'model', 'parts': ["Understood! I'll respond to all messages in this channel."]},
            ])
            message_history[chan_id] = model.start_chat(history=adjusted_history)
        else:
            message_history[chan_id] = model.start_chat(history=bot_template)
    await persist_data()
    await interaction.response.send_message("Bot personality updated!", ephemeral=True)

@bot.tree.command(name="clear_history", description="Clear conversation history for this channel.", guild=discord.Object(id=GUILD_ID))
async def clear_history(interaction: discord.Interaction):
    channel_id = interaction.channel.id
    await configure_gemini(channel_id)  # Reconfigure Gemini to update bot_template
    message_history[channel_id] = model.start_chat(history=bot_template)
    await persist_data()
    await interaction.response.send_message("Conversation history cleared for this channel.", ephemeral=True)

@bot.tree.command(name="show_modifiers", description="Show the current relevance modifiers.", guild=discord.Object(id=GUILD_ID))
async def show_modifiers(interaction: discord.Interaction):
    modifiers = await load_modifiers()
    if modifiers:
        await interaction.response.send_message("Current modifiers:\n" + "\n".join(modifiers), ephemeral=True)
    else:
        await interaction.response.send_message("No modifiers set.", ephemeral=True)

# ----------------------------
# Helper Functions
# ----------------------------

async def upload_to_pastee(code: str, title: str = "Code Snippet") -> Optional[str]:
    """Uploads code to Pastee (pastes.io) and returns the URL."""
    pastee_api_key = PASTEE_BEARER_TOKEN # Use the new Pastee API key
    if not pastee_api_key:
        logger.error("PASTEE_BEARER_TOKEN is not set. Cannot upload to Pastee.")
        return None
    try:
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {pastee_api_key}'
        }
        data = {
            'content': code,
            'status': '1',  # 1 for public/unlisted, check Pastee docs for options
            'expire': 'N',  # N for no expiration
            'title': title
        }
        response = requests.post('http://pastes.io/api/paste/create', headers=headers, data=data) # Using HTTP as per user's curl command
        if response.status_code == 200:
            try:
                response_json = response.json()
                if response_json.get("success"): # Check for the "success" key which indicates success
                    pastee_url = response_json.get("success").get("paste_url") # Get the URL from inside the "success" key
                    logger.info(f"Code uploaded to Pastee: {pastee_url}")
                    parsed_url = urlparse(pastee_url)
                    if parsed_url.netloc.endswith("pastes.io"):
                        paste_code = parsed_url.path.strip("/")
                        raw_url = f"https://pastes.io/raw/{paste_code}"
                        return raw_url
                    else:
                        logger.error(f"Unexpected Pastebin URL format: {pastebin_url}")
                        return pastebin_url  # Fallback to original URL
                else:
                    error_message = response_json.get("message", "Unknown Pastee error") # Get error message from JSON if available
                    logger.error(f"Pastee API Error: {error_message}, Full response: {response_json}")
                    return None
            except json.JSONDecodeError:
                logger.error(f"Pastee API response is not valid JSON: {response.text}")
                return None

        else:
            logger.error(f"Pastee upload failed with status code: {response.status_code}, Response Text: {response.text}") # Log response text
            return None
    except Exception as e:
        logger.error(f"Error uploading to Pastee: {e}")
        return None


async def upload_to_pastebin(code: str, title: str = "Code Snippet") -> Optional[str]:
    """Uploads code to Pastebin and returns the URL."""
    if not PASTEBIN_API_KEY:
        logger.error("PASTEBIN_API_KEY is not set. Cannot upload to Pastebin.")
        return None
    try:
        data = {
            'api_dev_key': PASTEBIN_API_KEY,
            'api_option': 'paste',
            'api_paste_code': code,
            'api_paste_private': '1',  # Unlisted
            'api_paste_name': title,
            'api_paste_format': 'python'  # or 'text', 'javascript', etc.
        }
        response = requests.post('https://pastebin.com/api/api_post.php', data=data)
        if response.status_code == 200:
            pastebin_url = response.text
            if "Bad API request" in pastebin_url:
                logger.error(f"Pastebin API Error: {pastebin_url}")
                return None
            logger.info(f"Code uploaded to Pastebin: {pastebin_url}")
            parsed_url = urlparse(pastebin_url)
            if parsed_url.netloc.endswith("pastebin.com"):
                paste_code = parsed_url.path.strip("/")
                raw_url = f"https://pastebin.com/raw/{paste_code}"
                return raw_url
            else:
                logger.error(f"Unexpected Pastebin URL format: {pastebin_url}")
                return pastebin_url  # Fallback to original URL
        else:
            logger.error(f"Pastebin upload failed with status code: {response.status_code}, Response Text: {response.text}") # Log response text
            return None
    except Exception as e:
        logger.error(f"Error uploading to Pastebin: {e}")
        return None


async def scrape_and_markdown(input_data: str, source: str = 'url') -> str:
    """
    Processes input data by scraping a URL or converting raw text to markdown.
    Parameters:
    - input_data (str): The URL to scrape or the raw text to process.
    - source (str): The type of input ('url' or 'text').
    Returns:
    - str: The processed markdown content or an error message.
    """
    try:
        if source == 'url':
            async with aiohttp.ClientSession() as session:
                async with session.get(input_data) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        extracted_text = trafilatura.extract(html_content)
                        if extracted_text:
                            return markdownify.markdownify(extracted_text)
                        else:
                            return f"Could not extract content from {input_data}"
                    else:
                        return f"Error: Could not retrieve content from {input_data} (Status: {response.status})"
        elif source == 'text':
            # Check if input is HTML or plain text
            if any(tag in input_data for tag in ("<html", "<body", "<p", "<div", "<span")):
                return markdownify.markdownify(input_data)  # Convert HTML to Markdown
            else:
                return http.escape(input_data)  # Escape plain text safely
        else:
            return "Invalid source type. Use 'url' or 'text'."
    except Exception as e:
        return f"Error processing input: {e}"

async def searx_search(query: str) -> List[str]:
    """
    Searches the web using SearxNG and returns a list of URLs.
    Uses an in-memory cache to avoid duplicate requests.
    """
    if query in searx_cache:
        logger.debug(f"Using cached search results for query: {query}")
        return searx_cache[query]
    base_url = "https://searx.irawrz.tv"
    search_url = f"{base_url}/search?q={query}&format=json"
    try:
        async with http_session.get(search_url) as response:
            if response.status == 200:
                data = await response.json()
                results = data.get("results", [])
                urls = [result.get("url") for result in results[:3] if result.get("url")]
                searx_cache[query] = urls
                return urls
            else:
                logger.error(f"Search request failed with status code {response.status}")
                return []
    except Exception as e:
        logger.error(f"Error occurred during the search: {e}")
        return []

async def send_long_message(channel: discord.abc.Messageable, text: str, is_code: bool = False):
    """
    Sends a long message to a Discord channel.
    - If the message is a long code block (over 20 lines), it uploads to Pastee.
    - Regular long messages are split into chunks.
    """
    max_length = 2000
    if is_code:
        # Extract the code from the message
        code_blocks = re.findall(r'```(?:python)?\n([\s\S]*?)```', text)
        if not code_blocks:
            code_blocks = [text]  # Fallback if no code blocks are found
        for code in code_blocks:
            lines = code.strip().split('\n')
            if len(lines) > 20:
                pastee_url = await upload_to_pastee(code, title="Code Snippet") # Try Pastee first
                if pastee_url:
                    await channel.send(f"Code snippet too long, uploaded to Pastee: {pastee_url}")
                    return # Stop after successful Pastee upload and sending message
                else:
                    pastebin_url = await upload_to_pastebin(code, title="Code Snippet (Pastee failed)") # Fallback to Pastebin
                    if pastebin_url:
                        await channel.send(f"Code snippet too long, uploaded to Pastebin (Pastee failed): {pastebin_url}")
                        return # Stop after successful Pastebin upload and sending message
                    else:
                        await channel.send("Code too long and failed to upload to Pastee and Pastebin. Check logs for errors.")
                        return # Stop after both Paste services fail
            else: # Send short code blocks directly
                if text: # Check if text is not empty after pastebin handling
                    await channel.send(text)
                return # Stop after sending short code block
    # Handle regular messages or short code blocks
    if len(text) > max_length:
        # Split the message intelligently based on sentence endings
        while len(text) > max_length:
            split_index = max(
                text.rfind('. ', 0, max_length),
                text.rfind('? ', 0, max_length),
                text.rfind('! ', 0, max_length)
            )
            if split_index == -1:
                split_index = max_length
            chunk = text[:split_index].strip()
            await channel.send(chunk)
            text = text[split_index:].strip()
    if text:
        await channel.send(text)

def reformat_links(text: str) -> str:
    """
    Replaces markdown-style links like ([link](link)) with just the URL.
    """
    pattern = r'\(\[(.*?)\]\((.*?)\)'
    return re.sub(pattern, r'\2', text)

async def get_channel_history(channel: discord.TextChannel, limit: int = 50) -> str:
    """
    Retrieves and formats the recent message history of a channel.
    """
    messages = []
    try:
        async for msg in channel.history(limit=limit):
            timestamp = msg.created_at.strftime('%I:%M %p')
            messages.append(f"**{msg.author.name} — {timestamp}**: {msg.content}")
        return "\n".join(reversed(messages))
    except discord.errors.Forbidden:
        return "I do not have permission to read this channel's history."
    except Exception as e:
        logger.error(f"Error getting channel history: {e}")
        return f"An error occurred while retrieving channel history: {e}"

async def is_direct_response(message: discord.Message) -> Tuple[bool, str]:
    """
    Determines if a message is a direct response based on context.
    Returns a tuple of (should_respond, reason).
    """
    channel_id = message.channel.id
    if bot.user.mentioned_in(message):
        return True, "Bot explicitly mentioned"
    if message.reference is not None:
        return True, "Message is a reply"
    if channel_id in last_bot_question:
        bot_question = last_bot_question[channel_id]
        bot_keywords = re.findall(r'\w+', bot_question.lower())
        user_message = message.content.lower()
        if any(keyword in user_message for keyword in bot_keywords):
            return True, "Answering bot's question"
    try:
        messages = [msg async for msg in message.channel.history(limit=3)]
        if len(messages) < 2:
            return False, "Not enough messages in history"
        bot_messages = [msg for msg in messages if msg.author == bot.user]
        if not bot_messages:
            return False, "Bot hasn't spoken recently"
        last_bot_message = bot_messages[-1]
        user_message = message.content.lower()
        topic_keywords = ["news", "us", "america", "going on", "happening"]
        if any(keyword in user_message for keyword in topic_keywords):
            return True, "Continuing conversation with topic keywords"
        modifiers = await load_modifiers()
        modifier_string = "\n".join(modifiers)
        prompt = (
            f"You are an AI assistant named Gemmy. When addressing a user, always use their Discord name.\n"
            f"Determine if message 2 (the user's message) is directly responding to message 1 (the bot's message) "
            f"given the conversation context. Respond with a single number between 0 and 100 and a brief reason, separated by a comma.\n"
            f"If your name Gemmy is said the score is always 100"
            f"Modifiers:\n{modifier_string}\n"
            f"Examples:\n"
            f"- Bot: \"What's the weather today?\" User: \"It's sunny.\" -> 95, Direct answer.\n"
            f"- Bot: \"I'm going to the store.\" User: \"Okay.\" -> 70, Acknowledgment.\n"
            f"- Bot: \"I'm going to the store.\" User: \"What time is it?\" -> 10, Unrelated.\n"
            f"Message 1 (Bot): **{last_bot_message.author.name} — {last_bot_message.created_at.strftime('%I:%M %p')}**: {last_bot_message.content}\n"
            f"Message 2 (User): **{message.author.name} — {message.created_at.strftime('%I:%M %p')}**: {message.content}\n"
            f"Response (0-100), Reason:"
        )
        response = await asyncio.to_thread(model.generate_content, prompt)
        full_response = response.text.strip()
        logger.debug(f"Direct response AI output: {full_response}")
        try:
            direct_response_score, reason = full_response.split(",", 1)
            direct_response_score = int(direct_response_score.strip())
            reason = reason.strip()
        except ValueError:
            logger.warning(f"AI returned invalid format: {full_response}")
            return False, "Invalid response format from AI"
        return direct_response_score >= 50, reason
    except Exception as e:
        logger.error(f"Error checking direct response: {e}")
        return False, f"Error: {e}"

async def get_ai_response(channel_id: int, content: str, user_name: str, message: Optional[discord.Message] = None, depth: int = 0):
    """
    Gets the AI's response and handles tool invocations (e.g. web search, history summarization).
    The 'depth' parameter prevents infinite recursion.
    """
    global last_bot_question
    if depth > MAX_AI_RESPONSE_DEPTH:
        return "I'm sorry, but I am having trouble processing that request."
    try:
        if channel_id not in message_history:
            message_history[channel_id] = model.start_chat(history=bot_template)
        # Occasionally include the user's name
        if random.random() < 0.3:
            enhanced_content = f"The user's name is {user_name}. Remember this for future interactions. {content}"
        else:
            enhanced_content = content
        initial_response = await asyncio.to_thread(message_history[channel_id].send_message, enhanced_content)
        ai_response = initial_response.text
        last_bot_question[channel_id] = ai_response
        # If the AI asks to read history, fetch and summarize channel history.
        if "READ_HISTORY" in ai_response and message is not None:
            logger.info("Retrieving channel history for summarization...")
            channel_history = await get_channel_history(message.channel)
            if "permission" in channel_history.lower():
                await send_long_message(message.channel, channel_history)
                return ai_response
            summary_prompt = (
                f"Summarize the following Discord conversation:\n\n{channel_history}\n\n"
                "Focus on the main topics discussed and any key decisions or agreements made. Provide a concise summary."
            )
            ai_response = await get_ai_response(channel_id, summary_prompt, user_name, message, depth=depth + 1)
        # If the AI asks for a web search, perform the search and scrape content.
        if "NEED_WEB_SEARCH" in ai_response and message is not None:
            query = ai_response.split("NEED_WEB_SEARCH ", 1)[1].strip()
            logger.info(f"Performing web search for query: {query}")
            urls = await searx_search(query)
            if urls:
                scraped_content = []
                for url in urls:
                    markdown_content = await scrape_and_markdown(url)
                    scraped_content.append(f"Content from {url}:\n{markdown_content}")
                all_content = "\n\n".join(scraped_content)
                if len(all_content) > 10000:
                    all_content = all_content[:10000] + "\n[Content truncated due to length]"
                # Get a new AI response using the web content.
                ai_response = await get_ai_response(channel_id, f"Web page content for '{query}':\n{all_content}", user_name, message, depth=depth + 1)
            else:
                ai_response = "No search results found."
        return ai_response
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return "An error occurred while processing your message."
    finally:
        if channel_id in last_bot_question:
            del last_bot_question[channel_id]

async def should_respond(message: discord.Message) -> bool:
    """
    Determines if the bot should respond to a message.
    - If the message is from the bot itself or mentions everyone, it is skipped.
    - If the message contains "Gemmy" (case-insensitive), it ALWAYS responds.
    - In direct messages or in tracked channels, the bot always responds.
    - If smart replies are enabled for the channel, the message is evaluated via AI.
    - Otherwise (smart replies off), the bot responds only if the literal text "gemmy" is present in the message.
    """
    if message.author == bot.user or message.mention_everyone:
        return False
    # ALWAYS respond if the bot's name is mentioned
    if "gemmy" in message.content.lower():
        return True
    # Always respond in DMs
    if isinstance(message.channel, discord.DMChannel):
        return True
    # Always respond in manually tracked channels
    if message.channel.id in tracked_channels:
        return True
    # If smart replies are on, use our AI evaluation
    if smart_replies_status.get(message.channel.id, False):
        respond, reason = await is_direct_response(message)
        logger.debug(f"Smart Reply decision: {respond} ({reason})")
        return respond
    else:
        # With smart replies off, only respond if "gemmy" is literally in the message (case-insensitive)
        return False  # Now, it only responds if smart replies are enabled or bot's name is mentioned

# ----------------------------
# Event Handlers
# ----------------------------

@bot.event
async def on_message(message: discord.Message):
    """Handles incoming messages."""
    if message.author == bot.user:
        return
    try:
        channel_id = message.channel.id
        content = message.content
        user_name = message.author.name

        if not await should_respond(message):
            return

        # Extract URLs from the message content
        urls = re.findall(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        if urls:
            # Scrape the first URL found and add the content to the AI prompt
            url_to_scrape = urls[0]
            logger.info(f"Scraping URL: {url_to_scrape}")
            try:  # Add a try-except block around the scraping
                # Check if it's a raw Pastebin URL
                parsed_url = urlparse(url_to_scrape)
                if parsed_url.netloc.endswith("pastebin.com") and parsed_url.path.startswith("/raw/"):
                    # Fetch the raw text from Pastebin
                    async with http_session.get(url_to_scrape) as response:
                        if response.status == 200:
                            scraped_content = await response.text() # just get the raw text
                            content = f"Content from {url_to_scrape}:\n{scraped_content}\n\nUser's original message: {content}"
                        else:
                            await message.channel.send(f"Error fetching from Pastebin: Status {response.status}")
                            return # Important: Exit if there's an error
                else:
                    scraped_content = await scrape_and_markdown(url_to_scrape, source='url')
                    content = f"Content from {url_to_scrape}:\n{scraped_content}\n\nUser's original message: {content}"
            except Exception as e:
                logger.error(f"Error processing URL: {e}")
                await message.channel.send(f"Error processing URL: {e}")
                return # Exit if there's an error

        # Reconfigure Gemini with the channel-specific personality before getting the AI response
        await configure_gemini(channel_id)

        async with message.channel.typing():
            ai_response = await get_ai_response(channel_id, content, user_name, message)
            ai_response = reformat_links(ai_response)
            # Determine if the AI response contains a code block
            is_code = bool(re.search(r'```(?:python)?\n[\s\S]*?```', ai_response, re.IGNORECASE))
            # Use the modified send_long_message without forcing Pastebin for short code blocks
            await send_long_message(message.channel, ai_response, is_code=is_code)

        await persist_data()

    except Exception as e:
        logger.exception(f"Error processing message: {e}")
        await message.channel.send("An error occurred while processing your message.")

@bot.event
async def on_ready():
    """Called when the bot is ready."""
    global http_session
    logger.info(f'Logged in as {bot.user}')
    # Initialize the global HTTP session
    if http_session is None:
        http_session = aiohttp.ClientSession()

    # Load personalities
    personalities = await load_personalities()

    # Configure Gemini for tracked channels
    for channel_id in tracked_channels:
        await configure_gemini(channel_id)

    await load_data()

    # Sync commands (only once)
    try:
        synced_commands = await bot.tree.sync(guild=discord.Object(id=GUILD_ID))
        logger.info(f"Synced {len(synced_commands)} command(s) to guild.")
    except Exception as e:
        logger.error(f"Error syncing commands: {e}")
        traceback.print_exc()

    # Start periodic data persistence every 5 minutes.
    periodic_persist.start()

@bot.event
async def on_close():
    """Clean up resources on shutdown."""
    global http_session
    if http_session:
        await http_session.close()

@tasks.loop(minutes=5)
async def periodic_persist():
    """Periodically persist data to disk."""
    await persist_data()

# ----------------------------
# Main entry point
# ----------------------------

if __name__ == "__main__":
    try:
        bot.run(DISCORD_BOT_TOKEN)
    except KeyboardInterrupt:
        logger.info("Bot is shutting down due to KeyboardInterrupt.")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
