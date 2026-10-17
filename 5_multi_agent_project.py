                            #   --Multi Agent Research Assistant--

#we are building a Multi-Agent Research Assistant a fully autonomous Al system that thinks,
#searches. reads and writes on its own. Instead of a single Al answering your question from memory, 
#we are deploying a team of specialized intelligent agents that collaborat borate together to produce a professional research report on any topic you give them. 
#The Search Agent goes out on the live internet and finds the most relevant and recent sources.
#The Reader Agent then dives deep into those sources, scraping and extracting meaningful content.
#The Writer Agent takes all that eathered intelligence and crafts a well-structured, detailed report.
#And finally the Critic Agent reviews the entire report, scores it and gives feedback just like a senior researcher reviewing a junior's work.'
# Every single agent is powered by a Larse Language Model, 
#connected through LangChain's modern LCEL pipeline, and orchestrated through a shared memor memory system that makes them work as one unifed brain.
#This is not a chatbot. This is not a simple O&A tool.
#This is a production-level agentic AI system the kind of architecture that top AI companies are actively building



#--------------info of in prject Do---------
# ==========================================================
#  MULTI-AGENT RESEARCH ASSISTANT
#  Ek autonomous AI system jo topic dene par khud search karta hai,
#  sources padhta hai, report likhta hai aur uski quality check karta hai.
# ==========================================================
#
#  MAIN FEATURES:
#  1. Live web research: jawab LLM ki memory se nahi, real-time
#     internet data (Tavily Search API) se aata hai.
#  2. 4 specialized agents:
#       - Search Agent : relevant aur recent sources dhoondhta hai
#       - Reader Agent : pages scrape karke key facts nikalta hai (BeautifulSoup)
#       - Writer Agent : structured report likhta hai (LangChain LCEL chain)
#       - Critic Agent : report ko score (1-10) aur feedback deta hai
#  3. Self-improving loop: score kam ho to Critic ka feedback Writer ko
#     wapas jata hai aur report dobara likhi jaati hai.
#  4. Shared memory: sab agents ek common memory share karte hain
#     (sources, notes, drafts, reviews, scores), "ek brain" ki tarah.
#  5. Agent activity log: kaun agent ne kab aur kya kiya, uski history.
#  6. Error handling + auto-retry: API rate limit ya tool failure par
#     system crash nahi hota, khud retry karta hai.
#  7. Report export: final report aur Critic review .md file mein
#     reports/ folder mein automatically save hoti hai.
#  8. Interactive CLI: terminal se koi bhi topic dekar baar-baar
#     research chalaya ja sakta hai.
#
#  WORKFLOW:
#  Topic -> Search -> Reader -> Writer <-> Critic -> Final Report
#                (sab Shared Memory se jude hue)
#
#  TECH STACK:
#  Python, LangChain (LCEL + agents), Groq (LLM), Tavily API,
#  BeautifulSoup, Rich, uv
# =========================================================================

import re 
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rich import print

load_dotenv()

# ---------- Tool 1: Tavily web search ----------
_tavily = TavilySearch(
    tavily_api_key=os.getenv("TAVILY_API_KEY"),   # यहाँ key साफ़ दिख रही है
    max_results=4,
    topic="general",
    search_depth="advanced",
)

@tool
def web_search(query: str) -> str:
    """Search the live web for recent and relevant sources on a topic.
    Returns a numbered list of results with title, URL and a short snippet.,Search the live web. You MUST pass a 'query' argument: a short search string
    like 'AI news today'. Returns a numbered list of results with title, URL and snippet."""""
    try:
        response = _tavily.invoke({"query": query})
    except Exception as e:
        return f"Search failed: {e}"

    results = response.get("results", [])
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] {r.get('title', 'No title')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )
    return "\n----\n".join(lines)


# ---------- Tool 2: BeautifulSoup scraper ----------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


@tool
def scrape_url(url: str) -> str:
    """Open a web page and extract its main readable text content.
    Use this to read the full content of a source URL."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"Could not scrape {url}: {e}"

    soup = BeautifulSoup(resp.text, "lxml")

    # kachra hatao
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    if not text:
        return f"No readable content found at {url}"

    # LLM ke context limit ke liye truncate
    return text[:1500]


# ---------- Quick test ----------
#if __name__ == "__main__":
#    print(web_search.invoke({"query": "latest developments in AI agents"}))
#    print("-" * 60)
   # print(scrape_url.invoke({"url": "https://en.wikipedia.org/wiki/Artificial_intelligence"})[:500])




# ---------- Shared LLM (Groq) ----------
model = init_chat_model("openai/gpt-oss-120b", model_provider="groq", temperature=0 , max_retries =6,)


# ---------- 1. Search Agent (LLM + web_search tool) ----------
search_agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=(
        "You are a Search Agent. Your job is to find the most relevant and "
        "recent sources for the given research topic.\n"
        "- Use the web_search tool (you may search 2-3 times with different queries).\n"
        "- Return a clean list of the 4-5 best sources.\n"
        "- For each source give: Title, URL, and a one-line summary.\n"
        "- Do not write the report. Only return the source list."
        "- Always call web_search with a single argument named 'query' (a plain search string).\n"
        "- Use the web_search tool only ONCE with the best query.\n"
    ),
)


# ---------- 2. Reader Agent (LLM + scrape_url tool) ----------
reader_agent = create_agent(
    model=model,
    tools=[scrape_url],
    system_prompt=(
        "You are a Reader Agent. You receive a list of source URLs.\n"
        "- Use the scrape_url tool to open the 3 most useful URLs.\n"
        "- If a page fails to scrape, skip it and move to the next one.\n"
        "- Extract only the key facts, numbers, dates and quotes.\n"
        "- Return the notes grouped by source, with the URL under each group.\n"
        "- Do not invent anything that is not in the scraped text.\n"
        "- Use the scrape_url tool to open only the 2 most useful URLs.\n"
    ),
)


# ---------- 3. Writer Agent (LCEL chain: prompt | model | parser) ----------
writer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional research report writer. "
     "Write a well-structured, detailed report using ONLY the research notes given. "
     "Keep the report under 600 words.\n"
     "Use this structure:\n"
     "# Title\n"
     "## Executive Summary\n"
     "## Introduction\n"
     "## Key Findings (with sub-headings)\n"
     "## Analysis\n"
     "## Conclusion\n"
     "## Sources (list of URLs)\n"
     "Be factual. Do not make up facts or links."),
    ("human",
     "Topic: {topic}\n\n"
     "Research notes:\n{research}\n\n"
     "Critic feedback to address (may be empty):\n{feedback}\n\n"
     "Write the full report now."),
])

writer_chain = writer_prompt | model | StrOutputParser()


# ---------- 4. Critic Agent (LCEL chain: prompt | model | parser) ----------
critic_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior researcher reviewing a junior's report. "
     "Be strict but fair. Judge accuracy, depth, structure, clarity and use of sources.\n"
     "Reply in EXACTLY this format:\n"
     "Score: <number from 1 to 10>/10\n"
     "Strengths:\n- ...\n"
     "Weaknesses:\n- ...\n"
     "Suggestions:\n- ..."),
    ("human",
     "Topic: {topic}\n\n"
     "Report to review:\n{report}"),
])

critic_chain = critic_prompt | model | StrOutputParser()


# ---------- Helper: agent se final text nikalna ----------
def run_agent(agent, message: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    return result["messages"][-1].content

# ---------- Shared Memory (sab agents ka ek common dimaag) ----------
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SharedMemory:
    topic: str = ""
    sources: str = ""                                   # Search Agent likhta hai
    notes: str = ""                                     # Reader Agent likhta hai
    drafts: list = field(default_factory=list)          # Writer ke har version
    reviews: list = field(default_factory=list)         # Critic ke har review
    scores: list = field(default_factory=list)          # Critic ke scores (numbers)
    log: list = field(default_factory=list)             # kaun agent ne kab kya kiya

    # ---- likhne ke helpers ----
    def add_log(self, agent: str, message: str):
        time = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{time}] {agent}: {message}")

    def save_sources(self, sources: str):
        self.sources = sources
        self.add_log("Search Agent", "sources save kiye")

    def save_notes(self, notes: str):
        self.notes = notes
        self.add_log("Reader Agent", "notes save kiye")

    def save_draft(self, report: str):
        self.drafts.append(report)
        self.add_log("Writer Agent", f"draft v{len(self.drafts)} likha")

    def save_review(self, review: str):
        self.reviews.append(review)
        score = self.extract_score(review)
        self.scores.append(score)
        self.add_log("Critic Agent", f"review diya, score = {score}/10")

    # ---- padhne ke helpers ----
    @property
    def latest_draft(self) -> str:
        return self.drafts[-1] if self.drafts else ""

    @property
    def latest_review(self) -> str:
        return self.reviews[-1] if self.reviews else ""

    @property
    def latest_score(self) -> float:
        return self.scores[-1] if self.scores else 0.0

    @staticmethod
    def extract_score(review: str) -> float:
        # "Score: 8/10" ya "Score: 7.5/10" se number nikalta hai
        match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", review)
        return float(match.group(1)) if match else 0.0

    def show_log(self):
        for line in self.log:
            print(line)



# ---------- Pipeline: Search -> Read -> Write -> Critic (rewrite loop) ----------
MAX_REVISIONS = 1       # Critic ke baad max kitni baar rewrite hoga
PASS_SCORE = 7        # is score ya usse upar ho to report approve


def run_pipeline(topic: str) -> SharedMemory:
    mem = SharedMemory(topic=topic)

    # Step 1: Search Agent
    print("[bold cyan]🔎 Search Agent chal raha hai...[/bold cyan]")
    sources = run_agent(search_agent, f"Find sources about: {topic}")
    mem.save_sources(sources)

    # Step 2: Reader Agent (memory se sources padhta hai)
    print("[bold cyan]📖 Reader Agent chal raha hai...[/bold cyan]")
    notes = run_agent(
        reader_agent,
        f"Read these sources and extract key facts:\n{mem.sources}",
    )
    mem.save_notes(notes)
     

    # Step 3 + 4: Writer <-> Critic loop
    feedback = ""
    for attempt in range(MAX_REVISIONS + 1):
        print(f"[bold green]✍️ Writer Agent (draft {attempt + 1}) likh raha hai...[/bold green]")
        report = writer_chain.invoke({
            "topic": mem.topic,
            "research": mem.notes,
            "feedback": feedback,
        })
        mem.save_draft(report)

        print("[bold yellow]🧐 Critic Agent review kar raha hai...[/bold yellow]")
        review = critic_chain.invoke({
            "topic": mem.topic,
            "report": mem.latest_draft,
        })
        mem.save_review(review)
        print(f"   Score: {mem.latest_score}/10")

        if mem.latest_score >= PASS_SCORE:
            print("[bold green]✅ Report approve ho gayi![/bold green]")
            break

        if attempt < MAX_REVISIONS:
            print("[bold red]🔁 Score kam hai, Writer ko feedback dekar rewrite...[/bold red]")
            feedback = mem.latest_review
        else:
            print("[bold magenta]⚠️ Max revisions ho gaye, latest draft final maani gayi.[/bold magenta]")

    return mem

# ---------- Part 6: Main (topic input + report save) ----------
from pathlib import Path
from rich.markup import escape


def save_report(mem: SharedMemory) -> Path:
    """Final report ko reports/ folder mein .md file ke roop mein save karta hai."""
    Path("reports").mkdir(exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", mem.topic).strip("_")[:50] or "report"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("reports") / f"{safe_name}_{stamp}.md"

    content = (
        f"{mem.latest_draft}\n\n"
        f"---\n\n"
        f"## Critic Review (Final Score: {mem.latest_score}/10)\n\n"
        f"{mem.latest_review}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def main():
    print("[bold]🤖 Multi-Agent Research Assistant[/bold]")
    print("Quit karne ke liye 'exit' likho.\n")

    while True:
        topic = input("📝 Research topic: ").strip()

        if topic.lower() in ("exit", "quit", "q"):
            print("Bye! 👋")
            break
        if not topic:
            print("[red]Topic khali hai, kuch likho.[/red]\n")
            continue

        try:
            mem = run_pipeline(topic)
        except Exception as e:
            print(f"[bold red]❌ Pipeline fail ho gayi:[/bold red] {escape(str(e))}\n")
            continue

        print("\n" + "=" * 60)
        print(escape(mem.latest_draft))
        print("=" * 60)
        print(escape(mem.latest_review))

        path = save_report(mem)
        print(f"\n[bold green]💾 Report save ho gayi:[/bold green] {path}")

        print("\n[dim]--- Agent Log ---[/dim]")
        for line in mem.log:
            print(f"[dim]{escape(line)}[/dim]")
        print()


if __name__ == "__main__":
    main()



                                          # vedant you are the best 