"""
llm_service.py — Gemini LLM Integration
========================================
Provides conversational movie recommendations by:
  1. Interpreting the user's mood/vibe
  2. Re-ranking ChromaDB candidates
  3. Generating warm, conversational responses
"""

import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL

# Configure the Gemini client
genai.configure(api_key=GEMINI_API_KEY)


def interpret_mood_and_recommend(user_mood: str, candidate_movies: list[dict]) -> dict:
    """
    Takes the user's mood text and a list of candidate movies from ChromaDB,
    then uses Gemini to re-rank and generate a conversational response.

    Args:
        user_mood: The user's free-text mood description.
        candidate_movies: List of dicts with 'title', 'tmdbId', 'tags' keys
                          (from ChromaDB query results).

    Returns:
        dict with keys:
            - 'response': The full conversational LLM text
            - 'recommended_titles': List of up to 5 movie titles picked by the LLM
    """
    # Build the movie context block for the prompt
    movie_context = "\n".join([
        f"  {i+1}. {m['title']} — {m['tags'][:300]}"
        for i, m in enumerate(candidate_movies)
    ])

    prompt = f"""You are a warm, knowledgeable movie expert who genuinely loves cinema — from Hollywood blockbusters to Tollywood and Bollywood gems.

The user described their mood as: "{user_mood}"

Here are {len(candidate_movies)} candidate movies from our database that are semantically close to their mood:
{movie_context}

YOUR TASK:
1. Pick the TOP 5 movies from the list above that BEST match the user's mood.
2. For each pick, write a brief (1–2 sentence) reason WHY it fits their mood.
3. Rate each match: ⭐ (weak) to ⭐⭐⭐⭐⭐ (perfect).

FORMATTING RULES:
- Start with a short, friendly opening line acknowledging their mood (1 sentence).
- List each movie as: **Movie Title** — Match: ⭐⭐⭐⭐⭐
  followed by the reason on the next line.
- End with a short closing line inviting them to ask for more.
- Use emojis naturally but don't overdo it.
- IMPORTANT: You MUST only recommend movies from the provided list. Do NOT invent movies.
- IMPORTANT: Output EXACTLY 5 movies, no more, no less.

ALSO: After your response, on a new line, output a machine-readable line in this exact format:
TITLES_JSON: ["Movie Title 1", "Movie Title 2", "Movie Title 3", "Movie Title 4", "Movie Title 5"]
"""

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    response_text = response.text

    # Parse out the recommended titles from the machine-readable line
    recommended_titles = _parse_titles(response_text, candidate_movies)

    # Remove the TITLES_JSON line from the display response
    display_text = "\n".join(
        line for line in response_text.split("\n")
        if not line.strip().startswith("TITLES_JSON:")
    ).strip()

    return {
        "response": display_text,
        "recommended_titles": recommended_titles
    }


def _parse_titles(response_text: str, candidates: list[dict]) -> list[str]:
    """
    Extract movie titles from the LLM response.
    First tries the TITLES_JSON line, then falls back to matching
    candidate titles found in the response text.
    """
    import json

    # Try to parse the TITLES_JSON line
    for line in response_text.split("\n"):
        if line.strip().startswith("TITLES_JSON:"):
            try:
                json_str = line.split("TITLES_JSON:", 1)[1].strip()
                titles = json.loads(json_str)
                if isinstance(titles, list) and len(titles) > 0:
                    return titles[:5]
            except (json.JSONDecodeError, IndexError):
                pass

    # Fallback: find candidate titles that appear in the response
    found = []
    for movie in candidates:
        title = movie['title']
        if title in response_text and title not in found:
            found.append(title)
        if len(found) >= 5:
            break

    return found if found else [m['title'] for m in candidates[:5]]
