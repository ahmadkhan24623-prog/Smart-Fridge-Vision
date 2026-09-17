"""
AI Recipe Generator Module using Google Gemini Vision API (`google-genai` SDK).
Processes user-uploaded fridge and food images to detect ingredients and generate custom,
Michelin-grade structured recipes with 4-stage cooking breakdowns and timers.
"""

import os
import io
import json
import logging
import re
from PIL import Image

logger = logging.getLogger("ai_generator")

# Supported Gemini Vision Models in priority order
MODELS_PRIORITY = [
    "gemini-3.6-flash",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]

def resolve_gemini_api_key(explicit_key=None):
    """
    Resolves the Gemini API Key from:
    1. Explicit parameter passed from UI
    2. Environment variable GEMINI_API_KEY
    3. Streamlit secrets if running inside Streamlit
    4. Local .env file if present
    """
    if explicit_key and str(explicit_key).strip():
        return str(explicit_key).strip()
    
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
        
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
        
    # Check .env file if present
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(base_dir, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
    except Exception:
        pass

    return ""

def is_gemini_available(api_key=None):
    """Checks whether the google-genai library and an API key are present."""
    key = resolve_gemini_api_key(api_key)
    if not key:
        return False
    try:
        from google import genai
        return True
    except ImportError:
        try:
            import google.generativeai
            return True
        except ImportError:
            return False

def get_genai_client(api_key=None):
    """Initializes and returns a Google GenAI client."""
    resolved_key = resolve_gemini_api_key(api_key)
    if not resolved_key:
        raise ValueError(
            "No Gemini API key provided. Set $env:GEMINI_API_KEY in PowerShell or paste your key in the sidebar."
        )
    
    try:
        from google import genai
        client = genai.Client(api_key=resolved_key)
        return client
    except ImportError:
        raise ImportError("The 'google-genai' package is not installed. Run: pip install google-genai pillow")

def test_gemini_connection(api_key=None):
    """
    Verifies that the provided or configured Gemini API key is valid and can reach the API.
    Returns:
        tuple: (success: bool, message: str)
    """
    resolved_key = resolve_gemini_api_key(api_key)
    if not resolved_key:
        return False, "No API key found. Please input your Gemini API key in the sidebar or set $env:GEMINI_API_KEY."
    
    try:
        client = get_genai_client(api_key=resolved_key)
        for model_name in MODELS_PRIORITY:
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents="Ping. Respond with 'OK'.",
                )
                if resp and resp.text:
                    return True, f"Connected to Google Gemini ({model_name}) successfully!"
            except Exception as e:
                err_msg = str(e)
                if "API_KEY_INVALID" in err_msg or "400" in err_msg:
                    return False, f"Invalid API Key: {err_msg}"
                continue
        return False, "Could not contact any Gemini model. Check network connectivity or API key quota."
    except Exception as e:
        return False, f"Connection test failed: {e}"

RECIPE_PROMPT = """
You are an Executive Master Chef and Computer Vision AI for an elite smart fridge culinary workstation.
Analyze this food/fridge photo carefully and generate a complete, authentic, Michelin-quality recipe based STRICTLY on the visible food items.

CRITICAL INSTRUCTIONS:
1. Examine the image with extreme visual precision. Detect ONLY the actual food items, grains, vegetables, proteins, or ingredients present in the image.
   - If the image shows RICE, GRAINS, PEAS, or VEGETABLES: The dish MUST be a Rice/Grain dish (e.g. Fragrant Vegetable Fried Rice, Herb Rice Pilaf, Garlic Rice, Risotto). DO NOT invent or hallucinate chicken, beef, or meat if it is not clearly present!
   - If the image shows CHICKEN: Generate a chicken dish.
   - If the image shows BEEF / STEAK: Generate a beef dish.
   - If the image shows FISH / SEAFOOD: Generate a seafood dish.
   - If the image shows EGGS: Generate an egg dish.
2. Select or design a gourmet, delicious dish that maximally utilizes the detected items.
3. You MUST respond with ONLY a valid, single JSON object (no surrounding markdown text outside the JSON, no markdown block quotes).

JSON SCHEMA:
{
  "title": "Dish Name (e.g. Fragrant Vegetable Fried Rice with Garden Mint & Peas)",
  "category": "Non-Veg" | "Vegetarian" | "Seafood" | "Breakfast" | "Fast & Easy",
  "prep_time": "10 mins",
  "cook_time": "15 mins",
  "servings": "2-3 servings",
  "difficulty": "Easy" | "Medium" | "Advanced",
  "calories": "380 kcal",
  "detected_ingredients": ["rice", "carrot", "garlic", "onion", "oil"],
  "primary_ingredient": "rice",
  "quantities": "2 cups Basmati Rice (cooked)|1 medium Carrot (diced)|1/2 cup Green Peas|3 cloves Garlic|2 tbsp Sesame/Cooking Oil",
  "instructions": "Step 1: Prep vegetables...|Step 2: Heat oil and aromatics in wok...|Step 3: Toss in cooked rice and season...|Step 4: Garnish with fresh herbs and serve...",
  "chef_tip": "Professional chef sensory checkpoint or secret advice for this dish.",
  "stages": [
    {
      "stage_num": 1,
      "title": "Mise en Place & Seasoning",
      "tab_lbl": "Prep & Season",
      "timer_secs": 180,
      "timer_str": "03:00",
      "flame": "❄️ Ambient / Prep Station",
      "instruction": "Prep ingredients, dice vegetables, and measure seasoning."
    },
    {
      "stage_num": 2,
      "title": "High-Heat Wok Searing",
      "tab_lbl": "Wok Sear",
      "timer_secs": 240,
      "timer_str": "04:00",
      "flame": "🔥 High Flame (200°C - 220°C)",
      "instruction": "Sizzle aromatics in smoking hot oil until intensely fragrant."
    },
    {
      "stage_num": 3,
      "title": "Grain Tossing & Flavor Infusion",
      "tab_lbl": "Toss & Glaze",
      "timer_secs": 180,
      "timer_str": "03:00",
      "flame": "♨️ Medium Flame (140°C)",
      "instruction": "Add rice and vegetables, tossing continuously for optimal wok hei flavor."
    },
    {
      "stage_num": 4,
      "title": "Resting & Plating Presentation",
      "tab_lbl": "Garnish & Serve",
      "timer_secs": 120,
      "timer_str": "02:00",
      "flame": "🍽️ Off Heat / Presentation",
      "instruction": "Garnish with fresh mint sprigs or herbs and serve piping hot."
    }
  ]
}
"""

def clean_json_response(raw_text):
    """Robustly extracts and parses JSON even if surrounded by markdown fences or text."""
    if not raw_text:
        raise ValueError("Empty response text from Gemini API")
    raw = raw_text.strip()
    
    # Check for markdown code fence ```json ... ``` or ``` ... ```
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if code_block:
        candidate = code_block.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass
            
    # Try direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass
        
    # Try finding first { and last }
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = raw[first_brace:last_brace + 1].strip()
        return json.loads(candidate)
        
    raise ValueError(f"Could not parse valid JSON from AI response: {raw[:160]}...")

def analyze_image_and_generate_recipe(pil_image, api_key=None, custom_notes=None):
    """
    Multimodal Vision Call:
    Takes a PIL Image, sends to Google Gemini Vision, and returns a structured recipe dictionary.
    Returns:
        tuple: (success: bool, recipe_dict_or_error_msg: dict|str)
    """
    try:
        client = get_genai_client(api_key=api_key)
    except Exception as e:
        return False, str(e)
    
    # Ensure RGB
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
        
    prompt = RECIPE_PROMPT
    if custom_notes:
        prompt += f"\n\nAdditional User Request/Preferences: {custom_notes}"

    # Prepare Image Part for google-genai
    try:
        from google.genai import types
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=90)
        img_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
        contents_payload = [img_part, prompt]
    except Exception:
        # Fallback to direct PIL Image
        contents_payload = [pil_image, prompt]

    last_error = None
    for model_name in MODELS_PRIORITY:
        try:
            logger.info(f"Calling Gemini Vision API using model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=contents_payload,
            )
            
            if response and response.text:
                recipe_data = clean_json_response(response.text)
                # Enrich with AI Metadata
                recipe_data["is_ai_generated"] = True
                recipe_data["ai_model"] = model_name
                return True, recipe_data
            else:
                last_error = f"Empty response text from {model_name}"
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Model {model_name} encountered error: {e}")
            continue

    return False, f"Gemini Vision API Error: {last_error}"

def generate_recipe_from_ingredients(ingredients_list, api_key=None, custom_notes=None):
    """
    Text-based Generation Call:
    Takes a list of confirmed ingredients and generates a custom recipe via Gemini.
    """
    try:
        client = get_genai_client(api_key=api_key)
    except Exception as e:
        return False, str(e)

    ing_str = ", ".join(ingredients_list) if isinstance(ingredients_list, list) else str(ingredients_list)
    prompt = f"{RECIPE_PROMPT}\n\nUser Pantry Ingredients: {ing_str}"
    if custom_notes:
        prompt += f"\nAdditional User Request: {custom_notes}"

    last_error = None
    for model_name in MODELS_PRIORITY:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt],
            )
            if response and response.text:
                recipe_data = clean_json_response(response.text)
                recipe_data["is_ai_generated"] = True
                recipe_data["ai_model"] = model_name
                return True, recipe_data
        except Exception as e:
            last_error = str(e)
            continue

    return False, f"Gemini API Error: {last_error}"
