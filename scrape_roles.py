import os
import pickle
import json
from datetime import datetime
from langgraph.func import entrypoint, task
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=api_key)

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import add_messages
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
)
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import time
import requests
from typing import List
from pydantic import BaseModel, Field

class QuantInternRole(BaseModel):
    title: str = Field(description="The exact title of the open internship position.")
    url: str = Field(description="The direct application URL link if available, otherwise the main page URL.")
    requirements: str = Field(description="Brief summary of key tech stack or degree requirements.")
    def __getitem__(self, item):
        return getattr(self, item)

class CompanyRolesReport(BaseModel):
    company_name: str
    company_url: str
    has_quant_internships: bool
    matching_roles: List[QuantInternRole]
    def __getitem__(self, item):
        return getattr(self, item)

def load_company_urls(filepath: str) -> List[dict]:
    """Reads a .txt file formatted with: Company Name, job posting page URL, and a blank newline."""
    companies = []
    if not os.path.exists(filepath):
        print(f"Error: The file {filepath} does not exist.")
        return companies

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Process lines in blocks of 3
    for i in range(0, len(lines), 3):
        if i < len(lines) and lines[i]:  # Ensure the company name isn't blank
            name = lines[i]
            url = lines[i+1] if (i+1) < len(lines) else ""
            
            if url:
                companies.append({"name": name, "url": url})
    return companies

def scrape_page_text(url: str) -> str:
    """Scrapes the visible text from a webpage using a headless browser."""
    # print(f"Scraping {url}...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",  # Removes the "webdriver" flag
                    "--no-sandbox",
                    "--disable-infobars",
                    "--window-size=1920,1080"
                ]
            )
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
                device_scale_factor=1,
                is_mobile=False,
                has_touch=False,
                locale="en-US",
                timezone_id="America/New_York"
            )
            page = context.new_page()
            page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            page.goto(url, wait_until="networkidle", timeout=15000)
            html_content = page.content()
            browser.close()
            
            soup = BeautifulSoup(html_content, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
                
            return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def analyze_with_gemini(company_name: str, company_url: str, page_text: str) -> CompanyRolesReport:
    """Uses Gemini API to extract quant roles matching the precise schema."""
    system_instruction = f"""
    You are a data extraction assistant. Analyze the text scraped from the careers page of {company_name}.
    Identify all open internship roles that match a 'Quant Intern' (Quantitative Researcher Intern,
    Quantitative Trader Intern, Quantitative Developer Intern, or Portfolio Manager Intern).

    If no matching undergraduate/graduate student internship roles are found, set has_quant_internships to false.
    """
    structured_model = model.with_structured_output(CompanyRolesReport)
    structured_response = structured_model.invoke([
        SystemMessage(content=system_instruction),
        HumanMessage(content=page_text)
    ])
    updated_response = structured_response.model_copy(
        update={"company_url": company_url}
    )
    return updated_response

def compare_results(old_data: List[dict], new_data: List[dict]):
    """Compares new scraping results against pickled data to identify new jobs."""
    print("\n=== DELTA ANALYSIS (NEW ROLES DETECTED) ===")
    
    # Map old data into a dictionary for O(1) lookups
    old_map = {item["company_name"]: item for item in old_data}
    new_openings_found = False

    for new_company in new_data:
        name = new_company["company_name"]
        new_titles = {role["title"] for role in new_company["matching_roles"]}
        
        # If the company was scraped in the previous run
        if name in old_map:
            old_titles = {role["title"] for role in old_map[name]["matching_roles"]}
            # Mathematical set difference to find purely unique additions
            added_roles = new_titles - old_titles
            
            if added_roles:
                new_openings_found = True
                print(f"\n NEW OPENINGS DETECTED AT {name.upper()}:")
                for role in new_company["matching_roles"]:
                    if role["title"] in added_roles:
                        print(f"  * {role['title']} -> Apply: {new_company['company_url']}")
        else:
            # Entirely new company added to openroles.txt since last run
            if new_titles:
                new_openings_found = True
                print(f"\n🆕 NEW COMPANY DETECTED: {name.upper()}:")
                for role in new_company["matching_roles"]:
                    print(f"  * {role['title']} -> Apply: {new_company['company_url']}")

# Augment the LLM with tools
tools = []
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

@task
def call_llm(messages: list[BaseMessage]):
    """LLM decides whether to call a tool or not"""
    return model_with_tools.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant tasked with determining the technologies used by a webpage"
            )
        ]
        + messages
    )

@task
def call_tool(tool_call: ToolCall):
    """Performs the tool call"""
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)

@entrypoint()
def agent(messages: list[BaseMessage]):
    model_response = call_llm(messages).result()

    while True:
        if not model_response.tool_calls:
            break

        # Execute tools
        tool_result_futures = [
            call_tool(tool_call) for tool_call in model_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_llm(messages).result()

    messages = add_messages(messages, model_response)
    return messages


def main():
    companies = load_company_urls("openroles.txt")
    print(f"Found {len(companies)} companies: {''.join([i['name'] for i in companies])}")
    final_results = []
    intern_results = []
    for _,company in enumerate(companies):
        print(f"[{_+1}/{len(companies)}] Processing {company['name']}...")
        raw_text = scrape_page_text(company["url"])
        try:
            response = analyze_with_gemini(company["name"], company["url"], raw_text)
            final_results.append(response)
            if response.has_quant_internships==True:
                intern_results.append(response)
            print(response)
        except Exception as e:
            print(f" -> Failed parsing with Gemini: {e}")
    with open("results.txt", "w") as f:
        for i in final_results:
            f.write(f"{i.company_name}: {str(i.has_quant_internships)}")
            f.write("\n")
    with open("intern_results.txt", "w") as f:
        for i in intern_results:
            f.write(f"{i.company_name}\n")
            if i.has_quant_internships:
                for j in i.matching_roles:
                    f.write(f"\t{j.title}\n")
                    f.write(f"\t\t{j.requirements}")
                    f.write("\n")
            f.write(f"{i.company_url}\n")
            f.write("\n")
    pickle_filename = "intern_results.pkl"
    if os.path.exists(pickle_filename):
        try:
            with open(pickle_filename, "rb") as f:
                historical_results = pickle.load(f)

            # Perform comparative analysis block
            compare_results(historical_results, final_results)
        except Exception as e:
            print(f"Failed to read historical pickle state: {e}")
    else:
        print("\nFirst execution tracked. Creating tracking state base layer...")

    # 3. Serialize current results to pickle file for the next run
    with open(pickle_filename, "wb") as f:
        pickle.dump(final_results, f)
    print(f"\nState saved securely to {pickle_filename}")


# for chunk in agent.stream(messages, stream_mode="updates"):
    # print(chunk)
    # print("\n")

main()
