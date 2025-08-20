import os
import json
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys
load_dotenv()
serpapi_API_KEY = os.environ.get("SERPAPI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


# === Helper: Generate search keywords ===
def search_words(query):
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "Based on the user prompt, provide concise search keywords to find the right Wikipedia page. Example: 'coffee wiki'."},
            {"role": "user", "content": query}
        ]
    )
    search_words = response.choices[0].message.content.strip()
    return search_words


# === Helper: Extract important sections from Wikipedia ===
def extract_important_sections(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    # Rules
    skip_sections = {"see also", "references", "further reading", "external links", "notes"}
    keep_keywords = {"history", "etymology", "biology", "culture", "overview", "background"}

    content = {}

    # Intro
    intro_paragraphs = []
    for p in soup.select("p"):
        if p.find_previous("h2"):
            break
        intro_paragraphs.append(p.get_text(strip=True))
    if intro_paragraphs:
        content["Introduction"] = " ".join(intro_paragraphs)

    # Loop through sections
    for h2 in soup.find_all("h2"):
        section_title = h2.get_text(strip=True)
        section_key = section_title.lower()

        if any(skip in section_key for skip in skip_sections):
            continue
        if not any(kw in section_key for kw in keep_keywords):
            continue

        paragraphs = []
        for elem in h2.next_elements:
            if elem.name == "h2":
                break
            if elem.name == "p":
                paragraphs.append(elem.get_text(strip=True))

        if paragraphs:
            content[section_title] = " ".join(paragraphs)

    return content


# === Main: Perform Google search, fetch Wikipedia, summarize ===
def google_search(query):
    params = {
        "engine": "google",
        "q": search_words(query),
        "api_key": serpapi_API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", [])

    wiki_link = None
    for result in organic_results:
        if result.get("source") == "Wikipedia":
            wiki_link = result["link"]
            break

    if not wiki_link:
        return "No relevant Wikipedia page found."

    wiki_data = extract_important_sections(wiki_link)
    wiki_data = json.dumps(wiki_data)

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "As a helpful study guide, summarize the given Wikipedia page for reference."},
            {"role": "user", "content": wiki_data}
        ]
    )
    wiki_summary = response.choices[0].message.content
    return wiki_summary
