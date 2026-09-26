"""The read-only tools a model may call, and the parser for its call syntax.

PROTOCOL.md fixes the five tools (``web_search``, ``read_file``,
``list_directory``, ``search_files``, ``file_info``), the arguments each takes,
the shape of what it returns, and the two spellings of a call the bridge
accepts: a ``<tool_call>{json}</tool_call>`` tag and a fenced ```` ```tool ````
block holding the same JSON. Everything in this module is read-only by
construction: there is no tool that writes, edits, moves, deletes, runs a
command or opens a URL other than the search providers.

Nothing here raises at the tool boundary. A refusal (a path outside the root),
a bad argument, a missing file, a network error and a timeout all come back as
``ToolResult(ok=False, …)`` so the model gets something it can react to and the
request still finishes.

The only network traffic is one HTTPS GET per search provider, in the order of
``SEARCH_PROVIDERS``, until one of them yields a parseable result. The HTML
providers are parsed with ``html.parser``; the last one reads Wikipedia's JSON
search API and then its article summaries — no dependencies beyond the standard
library, and no API key on any provider. A provider that is merely *challenged*
(a 202, a 429, or a 200 holding no usable results) is asked again a couple of
times before the ladder moves on, because that refusal is per-IP and time-based;
a hard failure — DNS, a refused connection, TLS, a timeout — moves on at once.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: PROTOCOL.md's bound on the tool loop: call -> execute -> continue happens at
#: most this many times in one request.
TOOL_ROUNDS = 4

#: The exact call format the system message shows the model.
CALL_FORMAT = '<tool_call>{"name": "web_search", "arguments": {"query": "…"}}</tool_call>'

#: `read_file`'s default truncation point, in bytes.
DEFAULT_MAX_BYTES = 20_000
#: `web_search`'s default result count, and the ceiling a request may ask for.
DEFAULT_MAX_RESULTS = 5
MAX_MAX_RESULTS = 25
#: `search_files` returns at most this many paths; the walk stops at the scan
#: limit so a pathological tree cannot make the tool unbounded.
SEARCH_LIMIT = 200
SEARCH_SCAN_LIMIT = 2_000
#: A file whose first bytes contain NUL is treated as binary and refused.
BINARY_SNIFF_BYTES = 8_192

#: `web_search` walks the provider ladder in `SEARCH_PROVIDERS`; each request
#: gets this long before that provider counts as a timeout and the next is
#: asked. There is no single endpoint constant: the ladder is the endpoint.
SEARCH_TIMEOUT = 15.0
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

#: The headers every search request carries. DuckDuckGo's refusal does not
#: depend on them — the same headers get 200 in one window and 202 in the next —
#: but a real browser's User-Agent keeps the endpoints from refusing on sight.
_SEARCH_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

#: A *challenge* is a transient refusal rather than a real failure: the two
#: statuses the endpoints throttle with, and a 200 carrying no usable results.
#: A challenged provider is asked this many extra times, pausing
#: CHALLENGE_BACKOFF times the attempt number (about 0.5 s, then 1.0 s) between
#: tries, because the refusal is per-IP and time-based and usually lifts.
CHALLENGE_RETRIES = 2
CHALLENGE_BACKOFF = 0.5
CHALLENGE_STATUSES = (202, 429)
#: The ceiling on one `web_search` call, retries included. A single request
#: still gets SEARCH_TIMEOUT, but the ladder as a whole stops here so a run of
#: slow endpoints cannot make one search take minutes.
SEARCH_BUDGET = 40.0

#: Wikipedia's summary endpoint: the clean extract and the canonical URL for one
#: article title (underscored and percent-encoded).
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
#: How much of that extract the model gets: a sentence or two, capped so one
#: long sentence cannot paste an article into the context window.
WIKIPEDIA_SNIPPET_SENTENCES = 2
WIKIPEDIA_SNIPPET_CHARS = 400

#: The sentence the model is handed when the whole ladder failed: the lookup did
#: not happen, so the answer must not come from the model's own knowledge.
SEARCH_FAILURE_INSTRUCTION = (
    "No web search was available for this query, so say that you could not look "
    "the answer up rather than answering from your own knowledge."
)

#: Both call spellings, matched over the whole text (never anchored to a line —
#: a model may open a call mid-sentence) and merged in source order.
_TAG_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_FENCE_CALL_RE = re.compile(
    r"```[ \t]*tool\b[ \t]*\r?\n?[ \t]*(.*?)```", re.IGNORECASE | re.DOTALL
)


class _ToolError(Exception):
    """A condition the model should see as `ok:false`, never a crash."""


@dataclass(frozen=True)
class ToolCall:
    """One call the model asked for, as parsed from its output."""

    name: str
    #: The raw `arguments` value: a dict for a well-formed call, anything else
    #: for a mis-argued one (which `execute` reports rather than crashing on).
    arguments: Any


@dataclass
class ToolResult:
    """What a tool returned: a one-line summary, and detail when there is some."""

    ok: bool
    summary: str
    detail: Optional[str] = None

    def model_text(self) -> str:
        """The text fed back to the model as the tool message."""
        if self.detail is not None:
            return self.detail
        return self.summary


# MARK: - Call parsing


def _calls_from_json(block: str) -> List[ToolCall]:
    """The calls a JSON object (or array of objects) describes, or none."""
    try:
        parsed = json.loads(block)
    except ValueError:
        return []
    items = parsed if isinstance(parsed, list) else [parsed]
    calls: List[ToolCall] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        call = ToolCall(name=name.strip(), arguments=item.get("arguments", {}))
        if call.arguments is None:
            call = ToolCall(name=call.name, arguments={})
        calls.append(call)
    return calls


def parse_calls(text: str) -> List[ToolCall]:
    """Every call in a model's output, in order, ignoring malformed JSON.

    Accepts ``<tool_call>{json}</tool_call>`` (case-insensitive, whitespace
    tolerant) and a fenced ```` ```tool ```` block holding the same JSON. Any
    other text is not a call: it is the answer.
    """
    if not isinstance(text, str) or not text:
        return []
    spans: List[Tuple[int, int, str]] = []
    for pattern in (_TAG_CALL_RE, _FENCE_CALL_RE):
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), match.group(1)))
    spans.sort(key=lambda span: (span[0], span[1]))

    calls: List[ToolCall] = []
    taken_until = -1
    for start, end, block in spans:
        if start < taken_until:  # a tag inside a fence, or vice versa
            continue
        taken_until = end
        calls.extend(_calls_from_json(block))
    return calls


def call_summary(call: ToolCall) -> str:
    """PROTOCOL.md's one-line rendering of a call, e.g. ``read_file("note.txt")``."""
    arguments = call.arguments
    if isinstance(arguments, dict) and len(arguments) == 1:
        rendered = json.dumps(next(iter(arguments.values())), ensure_ascii=False)
    else:
        rendered = json.dumps(arguments, ensure_ascii=False)
    return f"{call.name}({rendered})"


# MARK: - Argument helpers


def _int_argument(
    arguments: Dict[str, Any], key: str, default: int, minimum: int = 1, maximum: Optional[int] = None
) -> int:
    if key not in arguments or arguments[key] is None:
        return default
    raw = arguments[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise _ToolError(f"`{key}` must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise _ToolError(f"`{key}` must be an integer") from None
    if value < minimum:
        raise _ToolError(f"`{key}` must be at least {minimum}")
    return min(value, maximum) if maximum is not None else value


def _string_argument(arguments: Dict[str, Any], key: str) -> str:
    raw = arguments.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise _ToolError(f"`{key}` must be a non-empty string")
    return raw.strip()


def _iso8601(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _kind(path: Path) -> str:
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "other"


def _entry(path: Path) -> Dict[str, Any]:
    """One directory entry as PROTOCOL.md's `{name, kind, bytes, modified}`."""
    try:
        info = path.stat()
        modified = _iso8601(info.st_mtime)
        size = int(info.st_size) if path.is_file() else 0
    except OSError:
        return {"name": path.name, "kind": "other", "bytes": 0, "modified": ""}
    return {"name": path.name, "kind": _kind(path), "bytes": size, "modified": modified}


# MARK: - The search providers


class _DuckDuckGoParser(HTMLParser):
    """Pulls `{title, url, snippet}` out of a DuckDuckGo results page.

    Both keyless DuckDuckGo endpoints share this parser: their result markup
    differs only in class names. `html.duckduckgo.com` marks a title `result__a`
    and its snippet `result__snippet`; `lite.duckduckgo.com` uses `result-link`
    and `result-snippet` (on the snippet `<td>`, not on an anchor).
    """

    #: The classes a result's title link and its snippet element may carry.
    TITLE_CLASSES = ("result__a", "result-link")
    SNIPPET_CLASSES = ("result__snippet", "result-snippet")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: List[Dict[str, str]] = []
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag == "a" and any(name in classes for name in self.TITLE_CLASSES):
            self.results.append(
                {
                    "title": "",
                    "url": _unwrap_link(attributes.get("href") or ""),
                    "snippet": "",
                }
            )
            self._in_title = True
            self._in_snippet = False
        elif any(name in classes for name in self.SNIPPET_CLASSES) and self.results:
            self._in_title = False
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" or tag == "td":
            self._in_title = False
            self._in_snippet = False

    def handle_data(self, data: str) -> None:
        if self._in_title and self.results:
            self.results[-1]["title"] += data
        elif self._in_snippet and self.results:
            self.results[-1]["snippet"] += data


class _BraveParser(HTMLParser):
    """Pulls `{title, url, snippet}` out of a Brave Search results page.

    Brave server-renders every organic result as `<div class="snippet"
    data-pos="N" data-type="web">` holding the title — an `<a>` whose href is
    the result's own absolute URL, wrapping `<div class="title
    search-snippet-title">` — and then `<div class="generic-snippet">` with the
    snippet text. Blocks carrying another `data-type` (the video cluster, for
    one) and every link back to Brave itself are skipped, so all that survives
    are external results.

    Because the block is found by its `data-type` attribute rather than by the
    hashed `svelte-…` class names, a restyle of the page does not break it: a
    page whose results no longer match parses to nothing, which the ladder reads
    as this provider having failed.
    """

    #: The class marking a result block's title and its snippet.
    TITLE_CLASS = "search-snippet-title"
    SNIPPET_CLASS = "generic-snippet"
    #: The attribute and value identifying one organic result block.
    BLOCK_ATTRIBUTE = "data-type"
    BLOCK_VALUE = "web"

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: List[Dict[str, str]] = []
        self._depth = 0
        self._block: Optional[int] = None
        self._title: Optional[int] = None
        self._snippet: Optional[int] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self._depth += 1
        attributes = dict(attrs)
        if (
            tag == "div"
            and self._block is None
            and attributes.get(self.BLOCK_ATTRIBUTE) == self.BLOCK_VALUE
        ):
            self._block = self._depth
            self._title = None
            self._snippet = None
            self.results.append({"title": "", "url": "", "snippet": ""})
            return
        if self._block is None:
            return
        classes = (attributes.get("class") or "").split()
        if tag == "div":
            if self._title is None and self.TITLE_CLASS in classes:
                self._title = self._depth
            elif self._snippet is None and self.SNIPPET_CLASS in classes:
                self._snippet = self._depth
        elif tag == "a" and not self.results[-1]["url"] and "thumbnail" not in classes:
            # The title's own anchor; the thumbnail copy that follows it is not
            # a second result.
            self.results[-1]["url"] = attributes.get("href") or ""

    def handle_endtag(self, tag: str) -> None:
        if self._block is not None:
            if self._title == self._depth:
                self._title = None
            if self._snippet == self._depth:
                self._snippet = None
            if self._block == self._depth:
                self._block = None
                self._title = None
                self._snippet = None
        if self._depth:
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._block is None or not self.results:
            return
        if self._title is not None:
            self.results[-1]["title"] += data
        elif self._snippet is not None:
            self.results[-1]["snippet"] += data


def _unwrap_link(href: str) -> str:
    """DuckDuckGo wraps results as ``//duckduckgo.com/l/?uddg=<urlencoded>``."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parts = urllib.parse.urlsplit(href)
    query = urllib.parse.parse_qs(parts.query)
    target = query.get("uddg")
    if target:
        href = urllib.parse.unquote(target[0])
    return href


def _is_external(url: str, host: str) -> bool:
    """Whether `url` is a real result rather than a link back to the engine."""
    netloc = urllib.parse.urlsplit(url).netloc.lower()
    return bool(netloc) and netloc != host and not netloc.endswith("." + host)


def _parse_results(page: str, parser_class: type[HTMLParser]) -> List[Dict[str, str]]:
    parser = parser_class()
    parser.feed(page)
    parser.close()
    results: List[Dict[str, str]] = []
    for result in parser.results:
        title = html.unescape(result["title"]).strip()
        snippet = " ".join(html.unescape(result["snippet"]).split())
        url = result["url"].strip()
        if not title or not url:
            continue
        results.append({"title": title, "url": url, "snippet": snippet})
    return results


@dataclass(frozen=True)
class _SearchProvider:
    """One keyless search endpoint: its name, URL template and host."""

    name: str
    #: The endpoint, with `{query}` where the encoded query belongs (and, for
    #: Wikipedia, `{max_results}` for how many hits to ask for).
    template: str
    #: The parser for this endpoint's own HTML markup; `None` for the JSON API,
    #: which `kind` sends down a different path before any HTML is parsed.
    parser: Optional[type[HTMLParser]]
    #: The endpoint's own host: links back to it are chrome, not results.
    host: str
    #: How a response is read: "html" runs `parser` over the page, "wikipedia"
    #: reads the MediaWiki search API and then each hit's article summary.
    kind: str = "html"


#: The provider ladder `web_search` walks in order. Every entry is a plain
#: keyless HTTPS endpoint, so there is no key to configure and no dependency to
#: install. Order matters: DuckDuckGo answers a machine it has rate limited with
#: HTTP 202, and that is not an answer — the next provider is asked (after a
#: retry, if the refusal looks transient). Wikipedia is last and answers a
#: different question: it is a factual lookup source, not a general web index,
#: so a general query reaches it only when every search engine has refused.
SEARCH_PROVIDERS: Tuple[_SearchProvider, ...] = (
    _SearchProvider(
        "duckduckgo",
        "https://html.duckduckgo.com/html/?{query}",
        _DuckDuckGoParser,
        "duckduckgo.com",
    ),
    _SearchProvider(
        "duckduckgo-lite",
        "https://lite.duckduckgo.com/lite/?{query}",
        _DuckDuckGoParser,
        "duckduckgo.com",
    ),
    _SearchProvider(
        "brave",
        "https://search.brave.com/search?{query}",
        _BraveParser,
        "search.brave.com",
    ),
    _SearchProvider(
        "wikipedia",
        "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json"
        "&srsearch={query}&srlimit={max_results}",
        None,
        "en.wikipedia.org",
        "wikipedia",
    ),
)


@dataclass(frozen=True)
class _Attempt:
    """One request's outcome: the results, the real reason for none, and whether
    that reason is a transient challenge the ladder should retry.

    The reason is empty exactly when there is at least one result, and it is
    always what actually happened — an HTTP status, the socket's own error, or a
    200 that carried nothing usable. It is never a guess.
    """

    results: List[Dict[str, str]]
    reason: str
    challenged: bool


def _http_get(url: str) -> Tuple[int, str, str]:
    """One real HTTPS GET: its status, its reason phrase and its decoded body.

    A transport failure — DNS, a refused connection, TLS, a timeout — is raised
    as-is, for the caller to name. A non-2xx status is returned rather than
    raised, so a challenge (202/429) can be told from a hard failure.
    """
    request = urllib.request.Request(url, headers=_SEARCH_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=SEARCH_TIMEOUT) as response:
            status = int(getattr(response, "status", 0) or 0)
            reason = str(getattr(response, "reason", "") or "")
            charset = response.headers.get_content_charset() or "utf-8"
            return status, reason, response.read().decode(charset, "replace")
    except urllib.error.HTTPError as exc:
        charset = exc.headers.get_content_charset() if exc.headers else None
        body = exc.read().decode(charset or "utf-8", "replace")
        return int(exc.code), str(exc.reason or ""), body


def _transport_reason(exc: BaseException) -> str:
    """The real reason a request never produced a response."""
    if isinstance(exc, urllib.error.URLError):
        return f"unreachable: {exc.reason}"
    if isinstance(exc, TimeoutError):
        return f"timed out after {SEARCH_TIMEOUT:g}s"
    return f"{exc.__class__.__name__}: {exc}"


def _status_reason(status: int, reason: str) -> str:
    """``HTTP <status> <phrase> (the endpoint did not serve a results page)``."""
    phrase = f" {reason.strip()}" if reason.strip() else ""
    return f"HTTP {status}{phrase} (the endpoint did not serve a results page)"


def _first_sentences(text: str) -> str:
    """A Wikipedia extract's first sentence or two, collapsed and capped."""
    collapsed = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", collapsed)
    snippet = " ".join(sentences[:WIKIPEDIA_SNIPPET_SENTENCES]).strip()
    if len(snippet) <= WIKIPEDIA_SNIPPET_CHARS:
        return snippet
    cut = snippet[:WIKIPEDIA_SNIPPET_CHARS]
    head, _, _tail = cut.rpartition(" ")
    return (head or cut).rstrip(" ,;:") + "…"


def _wikipedia_summary(title: Any) -> Optional[Dict[str, str]]:
    """One article's `{title, url, snippet}`, or `None` when its summary is unusable.

    The URL is the summary's own canonical `content_urls.desktop.page`, so it is
    the article address Wikipedia itself publishes, never one this module built.
    """
    if not isinstance(title, str) or not title.strip():
        return None
    slug = urllib.parse.quote(title.strip().replace(" ", "_"))
    try:
        status, _reason, body = _http_get(WIKIPEDIA_SUMMARY_URL.format(title=slug))
    except OSError:
        return None
    if status != 200:
        return None
    try:
        summary = json.loads(body)
        extract = summary["extract"]
        page = summary["content_urls"]["desktop"]["page"]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(extract, str) or not extract.strip():
        return None
    if not isinstance(page, str) or not page:
        return None
    name = summary.get("title")
    return {
        "title": name if isinstance(name, str) and name else title,
        "url": page,
        "snippet": _first_sentences(extract),
    }


def _ask_wikipedia(
    provider: _SearchProvider, query: str, max_results: int
) -> _Attempt:
    """The MediaWiki search API's hits, each with its article's own summary."""
    url = provider.template.format(
        query=urllib.parse.quote_plus(query), max_results=max_results
    )
    try:
        status, reason, body = _http_get(url)
    except OSError as exc:
        return _Attempt([], _transport_reason(exc), False)
    if status != 200:
        return _Attempt([], _status_reason(status, reason), status in CHALLENGE_STATUSES)
    try:
        hits = json.loads(body)["query"]["search"]
        if not isinstance(hits, list):
            raise TypeError("`search` is not a list")
    except (ValueError, KeyError, TypeError):
        return _Attempt(
            [], "HTTP 200 but the response was not the search API's JSON", True
        )
    if not hits:
        # The API answered: no article matched. That is a real empty, not a
        # refusal, so the ladder moves on without retrying it.
        return _Attempt([], "no Wikipedia article matched the query", False)
    results: List[Dict[str, str]] = []
    for hit in hits:
        article = _wikipedia_summary(hit.get("title") if isinstance(hit, dict) else None)
        if article is not None:
            results.append(article)
            if len(results) >= max_results:
                break
    if not results:
        return _Attempt([], "HTTP 200 but no Wikipedia summary could be read", True)
    return _Attempt(results, "", False)


def _ask_html(provider: _SearchProvider, query: str) -> _Attempt:
    """One keyless HTML endpoint's real results, or the reason it gave none."""
    url = provider.template.format(query=urllib.parse.urlencode({"q": query}))
    try:
        status, reason, page = _http_get(url)
    except OSError as exc:
        return _Attempt([], _transport_reason(exc), False)
    if status != 200:
        return _Attempt([], _status_reason(status, reason), status in CHALLENGE_STATUSES)
    parser = provider.parser
    assert parser is not None, f"{provider.name} is an HTML provider without a parser"
    results = [
        result
        for result in _parse_results(page, parser)
        if _is_external(result["url"], provider.host)
    ]
    if not results:
        return _Attempt(
            [],
            "HTTP 200 but the page held no result link (the endpoint may be "
            "rate limiting)",
            True,
        )
    return _Attempt(results, "", False)


def _ask_provider(
    provider: _SearchProvider, query: str, max_results: int
) -> _Attempt:
    """One attempt at one provider, dispatched by how its endpoint answers."""
    if provider.kind == "wikipedia":
        return _ask_wikipedia(provider, query, max_results)
    return _ask_html(provider, query)


def _rendered_results(provider: _SearchProvider, results: List[Dict[str, str]]) -> str:
    """One provider's answer in the shape the model reads."""
    return "\n".join(
        [f"provider: {provider.name}"]
        + [
            f"{index}. {result['title']}\n   {result['url']}"
            + (f"\n   {result['snippet']}" if result["snippet"] else "")
            for index, result in enumerate(results, start=1)
        ]
    )


def _search(query: str, max_results: int) -> ToolResult:
    """Walk the provider ladder; the first real result answers.

    A challenged provider — a 202, a 429, or a 200 with nothing usable — is
    asked again up to CHALLENGE_RETRIES more times, with a short pause between
    tries, because that refusal is per-IP and time-based and usually lifts; a
    hard failure moves on at once. Every reason recorded is real and the
    attempts each provider got are reported, so when the whole ladder fails the
    model is told which provider said what, is never handed an empty success,
    and is told not to answer from its own knowledge instead.
    """
    started = time.monotonic()
    failures: List[str] = []
    for provider in SEARCH_PROVIDERS:
        if time.monotonic() - started >= SEARCH_BUDGET:
            failures.append(
                f"{provider.name}: not attempted (the {SEARCH_BUDGET:g}s search "
                "budget was already spent)"
            )
            continue
        attempts = 0
        while True:
            attempts += 1
            attempt = _ask_provider(provider, query, max_results)
            if not attempt.challenged or attempts > CHALLENGE_RETRIES:
                break
            pause = CHALLENGE_BACKOFF * attempts
            if time.monotonic() - started + pause >= SEARCH_BUDGET:
                break
            time.sleep(pause)
        if attempt.results:
            results = attempt.results[:max_results]
            return ToolResult(
                True,
                f"{len(results)} results",
                detail=_rendered_results(provider, results),
            )
        word = "attempt" if attempts == 1 else "attempts"
        failures.append(f"{provider.name}: {attempt.reason} after {attempts} {word}")

    return ToolResult(
        False,
        "every search provider failed",
        detail="\n".join(
            [f"web_search: no provider returned a result for {query!r}:"]
            + [f"- {failure}" for failure in failures]
            + ["", SEARCH_FAILURE_INSTRUCTION]
        ),
    )


# MARK: - The registry


@dataclass(frozen=True)
class _Spec:
    name: str
    #: One-line form for the hand-written fallback system message.
    signature: str
    description: str
    #: JSON-schema properties, for the tools the model is offered through its
    #: own chat template (the primary path).
    parameters: Dict[str, Any]
    required: Tuple[str, ...]


#: The five tools, in the order the model is shown them. This *is* the model's
#: view of what it may call; `execute` refuses anything else.
TOOL_SPECS: Tuple[_Spec, ...] = (
    _Spec(
        "web_search",
        "web_search(query: string, max_results: integer = 5) -> {results: [{title, url, snippet}]}",
        "Search the web. Use it for facts you cannot read from a file. The last "
        "provider is Wikipedia, a factual lookup source rather than a general "
        "web index: reach for it for a person, place or thing, not for "
        "site:-style or news queries.",
        {
            "query": {"type": "string", "description": "The search query."},
            "max_results": {
                "type": "integer",
                "description": "How many results to return.",
                "default": DEFAULT_MAX_RESULTS,
            },
        },
        ("query",),
    ),
    _Spec(
        "read_file",
        "read_file(path: string, max_bytes: integer = 20000) -> {path, bytes, text}",
        "Read a UTF-8 text file, truncated at max_bytes. Binary files are refused.",
        {
            "path": {"type": "string", "description": "The file to read."},
            "max_bytes": {
                "type": "integer",
                "description": "Truncate the text at this many bytes.",
                "default": DEFAULT_MAX_BYTES,
            },
        },
        ("path",),
    ),
    _Spec(
        "list_directory",
        "list_directory(path: string) -> {path, entries: [{name, kind, bytes, modified}]}",
        "List one directory. `kind` is file, directory or other; `modified` is ISO-8601.",
        {"path": {"type": "string", "description": "The directory to list."}},
        ("path",),
    ),
    _Spec(
        "search_files",
        "search_files(pattern: string, path: string = \".\") -> {matches: [paths]}",
        "Recursively glob for `pattern` (for example *.txt) under `path`, capped at 200 paths.",
        {
            "pattern": {
                "type": "string",
                "description": "A glob pattern, for example *.txt or **/*.md.",
            },
            "path": {
                "type": "string",
                "description": "The directory to search under.",
                "default": ".",
            },
        },
        ("pattern",),
    ),
    _Spec(
        "file_info",
        "file_info(path: string) -> {path, kind, bytes, modified}",
        "Describe one path without reading its contents.",
        {"path": {"type": "string", "description": "The path to describe."}},
        ("path",),
    ),
)


def tool_schemas() -> List[Dict[str, Any]]:
    """The tools in the OpenAI shape a chat template expects for `tools=`."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": {
                    "type": "object",
                    "properties": spec.parameters,
                    "required": list(spec.required),
                },
            },
        }
        for spec in TOOL_SPECS
    ]


class ToolRegistry:
    """The tool set, its descriptions and its path policy.

    Every path is resolved (symlinks included) and confined to ``root``:
    ``$SLAM_LM_TOOL_ROOT`` when set, the home directory otherwise. Nothing is
    ever written.
    """

    def __init__(self, root: Optional[Any] = None):
        configured = root if root is not None else os.environ.get("SLAM_LM_TOOL_ROOT")
        candidate = Path(str(configured)).expanduser() if configured else Path.home()
        try:
            self.root = candidate.resolve()
        except OSError:  # pragma: no cover - a broken $SLAM_LM_TOOL_ROOT
            self.root = Path.home().resolve()
        self._handlers: Dict[str, Any] = {
            "web_search": self._web_search,
            "read_file": self._read_file,
            "list_directory": self._list_directory,
            "search_files": self._search_files,
            "file_info": self._file_info,
        }

    # -- the model's instructions

    def system_prompt(self) -> str:
        """The hand-written fallback message.

        Used when the tokenizer has no chat template, or its template rejects
        the `tools=` argument, so the model is told the tools and PROTOCOL.md's
        call format in plain text. A tokenizer that takes `tools=` is offered
        the schemas by the template instead and never sees this.
        """
        lines = [
            "You are a helpful assistant running locally on the user's Mac.",
            "",
            "You may call these read-only tools before you answer:",
            "",
        ]
        for spec in TOOL_SPECS:
            lines.append(f"- {spec.signature}")
            lines.append(f"  {spec.description}")
        lines += [
            "",
            "To call a tool, reply with that single line and nothing else:",
            CALL_FORMAT,
            "A fenced ```tool block holding the same JSON is accepted too.",
            "",
            "The result comes back as a tool message; then answer the user in plain text.",
            "Never invent a result: if a call fails, react to the error it reports.",
            f"File paths are resolved inside {self.root}; anything outside it is refused.",
        ]
        return "\n".join(lines)

    # -- execution

    def execute(self, call: ToolCall) -> ToolResult:
        """Run one call. Never raises: every failure is an `ok:false` result."""
        handler = self._handlers.get(call.name)
        if handler is None:
            return ToolResult(
                False,
                f"unknown tool: {call.name}",
                detail=f"unknown tool {call.name!r}; the tools are: "
                + ", ".join(spec.name for spec in TOOL_SPECS),
            )
        if not isinstance(call.arguments, dict):
            return ToolResult(
                False,
                "`arguments` must be a JSON object",
                detail=f"{call.name}: `arguments` must be a JSON object, got "
                f"{type(call.arguments).__name__}",
            )
        try:
            return handler(call.arguments)
        except _ToolError as exc:
            return ToolResult(False, str(exc), detail=str(exc))
        except Exception as exc:  # a real failure, honestly reported
            message = f"{exc.__class__.__name__}: {exc}"
            return ToolResult(False, f"{call.name} failed: {message}", detail=message)

    # -- path policy

    def resolve(self, raw: Any, tool: str) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise _ToolError(f"{tool} needs a `path` string")
        candidate = Path(raw.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise _ToolError(f"could not resolve {raw!r}: {exc}") from None
        if not resolved.is_relative_to(self.root):
            raise _ToolError(
                f"{resolved} is outside the tool root ({self.root})"
            )
        return resolved

    def _inside(self, path: Path) -> bool:
        try:
            return path.resolve().is_relative_to(self.root)
        except (OSError, RuntimeError):  # a broken link, or a symlink loop
            return False

    # -- tools

    def _web_search(self, arguments: Dict[str, Any]) -> ToolResult:
        query = _string_argument(arguments, "query")
        max_results = _int_argument(
            arguments, "max_results", DEFAULT_MAX_RESULTS, minimum=1, maximum=MAX_MAX_RESULTS
        )
        return _search(query, max_results)

    def _read_file(self, arguments: Dict[str, Any]) -> ToolResult:
        path = self.resolve(arguments.get("path"), "read_file")
        max_bytes = _int_argument(arguments, "max_bytes", DEFAULT_MAX_BYTES)
        if path.is_dir():
            raise _ToolError(f"{path} is a directory, not a file")
        if not path.is_file():
            raise _ToolError(f"no such file: {path}")

        total = int(path.stat().st_size)
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
        truncated = total > max_bytes
        if b"\x00" in data[:BINARY_SNIFF_BYTES]:
            raise _ToolError(
                f"{path} looks binary ({total} bytes, NUL bytes in the first "
                f"{min(BINARY_SNIFF_BYTES, len(data))}); read_file only reads UTF-8 text"
            )

        raw = data[:max_bytes]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            # A multi-byte character cut in half by the cap is not a binary file.
            text = ""
            if truncated:
                for trim in (1, 2, 3):
                    if trim >= len(raw):
                        break
                    try:
                        text = raw[: len(raw) - trim].decode("utf-8")
                        break
                    except UnicodeDecodeError:
                        continue
            if not text:
                raise _ToolError(
                    f"{path} is not UTF-8 text: {exc}"
                ) from None

        summary = f"{total} bytes" + (f" (truncated to {max_bytes})" if truncated else "")
        detail = text
        if truncated:
            detail += f"\n\n[read_file: truncated to {max_bytes} of {total} bytes]"
        return ToolResult(True, summary, detail=detail)

    def _list_directory(self, arguments: Dict[str, Any]) -> ToolResult:
        path = self.resolve(arguments.get("path"), "list_directory")
        if not path.is_dir():
            raise _ToolError(f"{path} is not a directory")
        entries = [_entry(child) for child in sorted(path.iterdir(), key=lambda p: p.name)]
        detail = "\n".join(
            f"{entry['kind']}\t{entry['bytes']}\t{entry['name']}" for entry in entries
        )
        return ToolResult(True, f"{len(entries)} entries", detail=detail or "(empty)")

    def _search_files(self, arguments: Dict[str, Any]) -> ToolResult:
        pattern = _string_argument(arguments, "pattern")
        if os.path.isabs(pattern):
            raise _ToolError("`pattern` must be relative, not an absolute path")
        path = self.resolve(arguments.get("path", "."), "search_files")
        if not path.is_dir():
            raise _ToolError(f"{path} is not a directory")
        matches: List[str] = []
        try:
            for candidate in path.rglob(pattern):
                # A symlink inside the root may point outside it: skip those.
                if not self._inside(candidate):
                    continue
                matches.append(str(candidate))
                if len(matches) >= SEARCH_SCAN_LIMIT:
                    break
        except (OSError, ValueError) as exc:
            raise _ToolError(f"could not search {path}: {exc}") from None
        matches.sort()
        capped = matches[:SEARCH_LIMIT]
        detail = "\n".join(capped)
        if len(matches) > SEARCH_LIMIT:
            detail += f"\n\n[search_files: {len(matches)} matches, first {SEARCH_LIMIT} shown]"
        return ToolResult(True, f"{len(capped)} matches", detail=detail or "(no matches)")

    def _file_info(self, arguments: Dict[str, Any]) -> ToolResult:
        path = self.resolve(arguments.get("path"), "file_info")
        try:
            info = path.stat()
        except FileNotFoundError:
            raise _ToolError(f"no such file or directory: {path}") from None
        kind = _kind(path)
        size = int(info.st_size) if kind == "file" else 0
        modified = _iso8601(info.st_mtime)
        payload = {
            "path": str(path),
            "kind": kind,
            "bytes": size,
            "modified": modified,
        }
        detail = "\n".join(f"{key}: {value}" for key, value in payload.items())
        return ToolResult(True, f"{kind}, {size} bytes", detail=detail)
