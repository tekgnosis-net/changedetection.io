"""
LLM fallback plugin for price and restock info extraction.

When the built-in structured-metadata extraction (JSON-LD, microdata, OpenGraph)
fails to produce both a price and availability, this plugin is called as a last
resort.  It sends a trimmed, HTML-stripped version of the page to the configured
LLM and asks it to return a structured JSON answer.

The module-level `datastore` variable is injected at startup by
`inject_datastore_into_plugins()` in pluggy_interface.py.
"""
import json
import re
from loguru import logger
from changedetectionio.pluggy_interface import hookimpl
from changedetectionio.llm.evaluator import apply_local_token_multiplier

# Injected at startup by inject_datastore_into_plugins()
datastore = None

SYSTEM_PROMPT = (
    'You are an expert price and restock extraction utility. '
    'Your task is to analyse a product page and determine the price and stock status of the MAIN product only.\n\n'

    'AVAILABILITY — treat as "in stock":\n'
    '- Action buttons near the product: "Add to cart", "Add to basket", "Buy now", '
    '"Order now", "Purchase", "Import", "Add to bag", "Add to trolley", "In stock", '
    '"Available", "Ships in X days/weeks", "In store", "Pick up today".\n'
    '- "Pre-order" or "Reserve" — the item is orderable, treat as "in stock".\n'
    '- "Only X left", "Almost gone", "Low stock", "Limited availability" — still in stock.\n'
    '- "Request a quote" or "Contact us for pricing" — item is available, price is null.\n'
    '- IMPORTANT: Ignore cart/basket/bag links in the page HEADER or navigation bar '
    '(e.g. a shopping cart icon showing item count). That reflects what is already in '
    'the visitor\'s cart — it says nothing about whether THIS product is available.\n\n'

    'PRICE — what NOT to use:\n'
    '- A "$0.00" or "0" that appears near header/nav links such as "Login", "Wishlist", '
    '"Contact Us", "My Account" is an empty shopping-cart indicator, NOT the product price. '
    'Ignore it entirely — return null for price rather than 0 in this situation.\n'
    '- Only return 0 (free) when the page clearly states the product itself costs nothing '
    '(e.g. "Free", "Free download", "Price: $0").\n\n'

    'AVAILABILITY — treat as "out of stock":\n'
    '- "Out of stock", "Sold out", "Unavailable", "Currently unavailable", '
    '"Temporarily out of stock", "Discontinued", "No longer available", '
    '"Notify me when available", "Email me when back", "Join waitlist".\n\n'

    'AVAILABILITY — return null when uncertain:\n'
    '- The page asks the user to select a size, colour, or other variant first '
    '("Select an option", "Choose a size") — availability depends on the variant, so return null.\n'
    '- You cannot clearly tell from the page content whether the item is available.\n\n'

    'PRICE rules:\n'
    '- Extract the main selling price as a plain number, no currency symbol.\n'
    '- Prices may use any popular locale format — interpret them all correctly and return a plain decimal number. '
    'Examples: "10 000 Kč" = 10000, "1.299,95 €" = 1299.95, "1,299.95" = 1299.95, '
    '"10 000,50" = 10000.50, "£1.299" = 1299, "¥10000" = 10000.\n'
    '- If both an original (crossed-out) price and a sale/current price appear, use the sale price.\n'
    '- "From $X" or "Starting at $X" are teaser prices — prefer a definite price or return null.\n'
    '- A price of 0 (free) is valid — return 0, not null.\n'
    '- If pricing requires a quote or login, return null for price.\n'
    '- Ignore prices shown in search/filter UI elements (e.g. "Price from: — to:").\n'
    '- IMPORTANT: Ignore ALL prices that appear inside or below recommendation/discovery blocks '
    'such as: "Similar items", "You may also like", "Customers also bought", '
    '"Based on your browsing", "Based on your shopping", "Frequently bought together", '
    '"People also viewed", "Related products", "Sponsored products", "More like this", '
    '"Other sellers", "Compare with similar items". '
    'These sections contain prices for OTHER products, not the main product.\n'
    '- When multiple prices appear on the page, prefer the price that is positioned '
    'earliest/highest in the page content — it is almost always the main product price. '
    'Prices appearing after large blocks of descriptive text or review sections are '
    'likely from recommendation widgets and should be ignored.\n\n'

    'CLASSIFIEDS AND LISTING PAGES:\n'
    '- On classifieds or marketplace sites (e.g. eBay listings, Craigslist, Bazoš, Gumtree), '
    'if a price is shown alongside seller contact details or a "Contact seller" link, '
    'treat the item as "instock" — the listing being active means it is available.\n\n'

    'Return ONLY a JSON object with exactly these three keys:\n'
    '  "price"        — number or null\n'
    '  "currency"     — ISO-4217 code (USD, EUR, GBP …) or null\n'
    '  "availability" — exactly one of: "instock", "outofstock", or null\n'
    '                   Use "instock" when the product can be ordered/purchased.\n'
    '                   Use "outofstock" when it cannot.\n'
    '                   Use null when you genuinely cannot tell.\n'
    'No markdown, no backticks, no explanation — pure JSON only.'
)

VISION_CUES_PROMPT = """VISUAL CUES (the page screenshot is provided alongside this text):
- A price with a strikethrough (a line drawn through it) is the ORIGINAL or REGULAR price — do NOT use it as the current price.
- A price displayed in a different colour (often red or orange), a larger font, or bolder weight than nearby prices is typically the current SALE/discounted price — prefer this as `price`.
- "SALE", "CLEARANCE", "%OFF", "WAS $X NOW $Y", "MEMBER PRICE", "CLUB" banners or badges indicate promotional pricing — treat these as visual confirmation that a sale is active.
- Use the screenshot to disambiguate when the page text alone is ambiguous about which of two prices is the current selling price."""

EXTRAS_INSTRUCTION_TEMPLATE = """ADDITIONALLY, the user has requested these extra data points:
{user_directive}

Include any extracted values as additional top-level keys in the JSON response, alongside `price`, `currency`, `availability`. Use lowercase snake_case keys (e.g. `sale_active`, `original_price`, `discount_percent`, `sale_label`, `urgency_text`). Values may be string, number, boolean, or null. Omit keys you cannot determine — do not return null for everything just to pad the schema."""


def build_system_prompt(use_vision: bool = False, extras_directive: str = '') -> str:
    """Compose the system prompt for the restock LLM call.

    The base SYSTEM_PROMPT carries the price/currency/availability rules.
    VISION_CUES_PROMPT prepends visual-cue guidance when the screenshot is
    being attached. EXTRAS_INSTRUCTION_TEMPLATE appends a user-defined
    extraction directive when one is set on the watch.
    """
    parts = []
    if use_vision:
        parts.append(VISION_CUES_PROMPT)
    parts.append(SYSTEM_PROMPT)
    if extras_directive and extras_directive.strip():
        parts.append(EXTRAS_INSTRUCTION_TEMPLATE.format(user_directive=extras_directive.strip()))
    return '\n\n'.join(parts)


_MAX_CONTENT_CHARS = 8_000


def _extract_jsonld(html_content: str) -> str:
    """Extract JSON-LD blocks — these contain reliable structured product data."""
    blocks = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html_content, flags=re.DOTALL | re.IGNORECASE
    )
    if not blocks:
        return ''
    combined = ' '.join(b.strip() for b in blocks)
    return combined[:2000]


# Semantic tags always treated as chrome (nav/header/footer)
_CHROME_TAGS = {'nav', 'header', 'footer', 'aside'}

# id/class fragments that strongly indicate navigation or site-chrome
_CHROME_PATTERNS = re.compile(
    r'\b(nav|navigation|navbar|menu|mega-menu|breadcrumb|breadcrumbs?|'
    r'site-header|page-header|top-bar|top-nav|top-header|mobile-nav|header-bar|'
    r'site-footer|page-footer|footer-links|related|similar|'
    r'you-?may-?also|customers?-?also|frequently-?bought|'
    r'people-?also|sponsored|recommendation|widget|sidebar|'
    r'cross-?sell|up-?sell)\b',
    re.IGNORECASE,
)


def _remove_chrome(html_content: str) -> str:
    """Use BS4 to strip navigation, header, footer and recommendation noise.

    Uses html.parser (built-in, no lxml) to avoid memory leak issues.
    Falls back to the original HTML string if BS4 fails for any reason.
    """
    try:
        from bs4 import BeautifulSoup, Tag
        soup = BeautifulSoup(html_content, 'html.parser')

        # Snapshot the full tag list before any decompositions so we don't
        # mutate the tree while iterating it.  After a parent is decomposed
        # its children become orphans (parent=None) — skip those.
        for tag in list(soup.find_all(True)):
            if not isinstance(tag, Tag) or tag.parent is None:
                continue
            name = tag.name or ''
            if name in _CHROME_TAGS:
                tag.decompose()
                continue
            try:
                cls_list = tag.get('class') or []
                cls_str = ' '.join(cls_list) if isinstance(cls_list, list) else str(cls_list)
                id_str = tag.get('id') or ''
            except Exception:
                continue
            if _CHROME_PATTERNS.search(cls_str + ' ' + id_str):
                tag.decompose()

        return str(soup)
    except Exception as e:
        logger.debug(f"BS4 chrome removal failed ({e}), using raw HTML")
        return html_content


def _strip_html(html_content: str) -> str:
    """HTML-to-text for LLM consumption.

    1. Extracts JSON-LD (structured product data) to prepend.
    2. Strips nav/header/footer/recommendation blocks via BS4.
    3. Removes all remaining tags and collapses whitespace.
    JSON-LD is prepended so reliable price/availability data is always visible
    to the LLM regardless of how deep it sits in the page.
    """
    jsonld = _extract_jsonld(html_content)

    # Remove site-chrome before generic tag stripping
    cleaned = _remove_chrome(html_content)

    # Drop HTML comments (can contain large disabled markup blocks)
    text = re.sub(r'<!--.*?-->', ' ', cleaned, flags=re.DOTALL)
    # Drop all <script> and <style> blocks
    text = re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common entities
    text = (text
            .replace('&nbsp;', ' ')
            .replace('&amp;', '&')
            .replace('&lt;', '<')
            .replace('&gt;', '>')
            .replace('&quot;', '"')
            .replace('&#39;', "'"))
    text = re.sub(r'\s+', ' ', text).strip()

    if jsonld:
        budget = _MAX_CONTENT_CHARS - len(jsonld) - 1
        return (jsonld + ' ' + text[:budget]).strip()
    return text[:_MAX_CONTENT_CHARS]


def run_llm_restock_extraction(watch, text_content, llm_intent=None):
    """LLM fallback for restock price/stock extraction (per-watch path).

    Two gates beyond the legacy global toggle:
      1. Per-watch llm_use_for_restock override (TernaryNoneBoolean: None=inherit global,
         True=force on, False=force off).
      2. Vision sub-gate (llm_use_vision + llm_vision_verified + provider_kind=='openai_compatible').
         Falls through to text-only on screenshot absence or completion failure;
         tracks 3-strikes via vision.record_vision_failure.

    Returns the parsed JSON dict (e.g. {'price': 10.0, 'currency': 'USD', 'availability': 'instock'})
    or None on any failure path.

    Token bookkeeping (global accumulate + per-watch counters) is handled internally,
    matching the pattern in evaluator.summarise_change.
    """
    if datastore is None:
        logger.debug("llm_restock: no datastore injected yet, skipping")
        return None

    from changedetectionio.llm.evaluator import (
        get_llm_config, accumulate_global_tokens, apply_local_token_multiplier
    )
    from changedetectionio.llm import client as llm_client
    from changedetectionio.llm import vision as _vision

    # Gate 1: per-watch override (TernaryNoneBoolean: None means inherit)
    per_watch = watch.get('llm_use_for_restock')
    if per_watch is True:
        effective = True
    elif per_watch is False:
        effective = False
    else:
        effective = bool(datastore.data['settings']['application']
                         .get('llm_restock_use_fallback_extract', True))

    if not effective:
        logger.debug("llm_restock: skipped (per-watch llm_use_for_restock=False or global disabled)")
        return None

    llm_cfg = get_llm_config(datastore)
    if not llm_cfg or not llm_cfg.get('model'):
        return None

    url = watch.get('url', '')
    user_prompt = f'URL: {url or "unknown"}\n\nPage content:\n{text_content}'
    if llm_intent:
        user_prompt += f'\n\nUser notification intent: {llm_intent}'

    # Compute extras directive up front (used in both vision and text-only prompts)
    extras_directive = (watch.get('llm_extract_extras') or '').strip()

    # Pre-build both system prompt variants so the right one is used at each call site
    system_prompt_text = build_system_prompt(use_vision=False, extras_directive=extras_directive)
    system_prompt_vision = build_system_prompt(use_vision=True, extras_directive=extras_directive)

    # Gate 2: vision sub-gate
    use_vision = (
        watch.get('llm_use_vision')
        and watch.get('llm_vision_verified')
        and llm_cfg.get('provider_kind') == 'openai_compatible'
    )

    messages = None
    if use_vision:
        loaded = _vision.load_and_prepare_screenshot(watch, llm_cfg)
        if loaded is not None:
            image_bytes, mime = loaded
            messages = _vision.build_vision_messages(
                text_user_content=user_prompt,
                image_bytes=image_bytes,
                mime_type=mime,
                system_prompt=system_prompt_vision,
            )

    if messages is None:
        messages = [
            {'role': 'system', 'content': system_prompt_text},
            {'role': 'user', 'content': user_prompt},
        ]

    _vision_was_used = any(isinstance(m.get('content'), list) for m in messages)

    def _bookkeep_tokens(_tokens, _input_tokens, _output_tokens):
        accumulate_global_tokens(
            datastore, _tokens,
            input_tokens=_input_tokens,
            output_tokens=_output_tokens,
            model=llm_cfg['model'],
        )
        if _tokens:
            watch['llm_last_tokens_used'] = _tokens
            watch['llm_tokens_used_cumulative'] = (watch.get('llm_tokens_used_cumulative') or 0) + _tokens

    try:
        raw, tokens, input_tokens, output_tokens = llm_client.completion(
            model=llm_cfg['model'], messages=messages,
            api_key=llm_cfg.get('api_key'), api_base=llm_cfg.get('api_base'),
            max_tokens=apply_local_token_multiplier(80, llm_cfg),
        )
        _bookkeep_tokens(tokens, input_tokens, output_tokens)
        if _vision_was_used:
            _vision.reset_vision_failure_count(watch)
    except Exception as e:
        logger.warning(f"llm_restock: completion failed: {e}")
        if _vision_was_used:
            _vision.record_vision_failure(watch)
            text_messages = [
                {'role': 'system', 'content': system_prompt_text},
                {'role': 'user', 'content': user_prompt},
            ]
            try:
                raw, tokens, input_tokens, output_tokens = llm_client.completion(
                    model=llm_cfg['model'], messages=text_messages,
                    api_key=llm_cfg.get('api_key'), api_base=llm_cfg.get('api_base'),
                    max_tokens=apply_local_token_multiplier(80, llm_cfg),
                )
                _bookkeep_tokens(tokens, input_tokens, output_tokens)
            except Exception as e2:
                logger.warning(f"llm_restock: text-only retry also failed: {e2}")
                return None
        else:
            return None

    raw = raw.strip()
    if raw.startswith('```'):
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = raw.rstrip('`').strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"llm_restock: JSON parse failed for raw={raw!r}: {e}")
        return None

    # Normalise price to float (preserve the legacy normalisation behavior)
    price = result.get('price')
    if price is not None:
        try:
            if isinstance(price, str):
                price = float(re.sub(r'[^\d.]', '', price))
            else:
                price = float(price)
            result['price'] = price
        except (ValueError, TypeError):
            logger.warning(f"llm_restock: could not convert price {price!r} to float, ignoring")
            result['price'] = None

    # Separate core keys from user-requested extras; persist extras to watch
    core_keys = {'price', 'currency', 'availability'}
    extras = {k: v for k, v in result.items() if k not in core_keys}
    # Reset to a fresh dict per run so stale extras from a previous run don't leak through
    watch['llm_extracted_extras'] = extras

    if result.get('price') is None and not result.get('availability'):
        logger.info(f"llm_restock: LLM returned no usable price or availability for {url}")
        return None

    return result


@hookimpl
def get_itemprop_availability_override(content, fetcher_name, fetcher_instance, url, llm_intent=None, watch=None):
    """Use an LLM as a last-resort fallback for price and restock extraction.

    When `watch` is provided, delegates to run_llm_restock_extraction (per-watch
    gates including llm_use_for_restock and vision support).  When `watch` is None
    (legacy callers, 3rd-party plugins testing the hook directly), uses the global
    llm_restock_use_fallback_extract flag and the existing text-only flow.
    """
    global datastore

    if datastore is None:
        logger.debug("LLM restock fallback: no datastore injected yet, skipping")
        return None

    # New path: per-watch gates via run_llm_restock_extraction
    if watch is not None:
        text_content = _strip_html(content) if content else ''
        if not text_content.strip():
            logger.debug("LLM restock fallback: no text content after stripping HTML")
            return None
        result = run_llm_restock_extraction(watch, text_content, llm_intent=llm_intent)
        if result is None:
            return None
        # processor.py expects {price, currency, availability}; tokens were tracked
        # internally by run_llm_restock_extraction so no _tokens/_model keys.
        return {
            'price': result.get('price'),
            'currency': result.get('currency') or None,
            'availability': result.get('availability') or None,
        }

    # ---- Legacy path (watch is None) ---- existing body unchanged from here ----

    # Gate on the user setting (default True — enabled out of the box)
    app_settings = datastore.data.get('settings', {}).get('application', {})
    if not app_settings.get('llm_restock_use_fallback_extract', True):
        logger.debug("LLM restock fallback: disabled in settings")
        return None

    try:
        from changedetectionio.llm.evaluator import get_llm_config, accumulate_global_tokens
        from changedetectionio.llm import client as llm_client
    except ImportError as e:
        logger.debug(f"LLM restock fallback: LLM libraries not available ({e})")
        return None

    llm_cfg = get_llm_config(datastore)
    if not llm_cfg or not llm_cfg.get('model'):
        logger.debug("LLM restock fallback: no LLM model configured, skipping")
        return None

    text_content = _strip_html(content) if content else ''
    logger.debug(f"LLM restock fallback: stripped HTML to {len(text_content)} chars for {url}")
    if not text_content.strip():
        logger.debug("LLM restock fallback: no text content after stripping HTML")
        return None

    logger.info(f"LLM restock fallback: using LLM ({llm_cfg['model']}) for price/stock extraction - {url}")

    user_prompt = f'URL: {url or "unknown"}\n\nPage content:\n{text_content}'
    if llm_intent:
        user_prompt += f'\n\nUser notification intent: {llm_intent}'

    try:
        raw, tokens, input_tokens, output_tokens = llm_client.completion(
            model=llm_cfg['model'],
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt},
            ],
            api_key=llm_cfg.get('api_key'),
            api_base=llm_cfg.get('api_base'),
            # 80 fits a {price, currency, availability} JSON answer comfortably for cloud
            # models. Local reasoning models burn most of that on chain-of-thought before
            # the JSON lands — the multiplier scales it up only when provider_kind says so.
            max_tokens=apply_local_token_multiplier(80, llm_cfg),
        )

        accumulate_global_tokens(
            datastore, tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=llm_cfg['model'],
        )

        # Strip optional markdown fences the model might add
        raw = raw.strip()
        if raw.startswith('```'):
            raw = re.sub(r'^```[a-z]*\n?', '', raw)
            raw = raw.rstrip('`').strip()

        logger.debug(f"LLM restock fallback raw response: {raw!r}")

        result = json.loads(raw)

        price = result.get('price')
        currency = result.get('currency') or None
        availability = result.get('availability') or None

        # Normalise price to float
        if price is not None:
            try:
                if isinstance(price, str):
                    price = float(re.sub(r'[^\d.]', '', price))
                else:
                    price = float(price)
            except (ValueError, TypeError):
                logger.warning(f"LLM restock fallback: could not convert price {price!r} to float, ignoring")
                price = None

        if price is None and not availability:
            logger.info(f"LLM restock fallback: LLM returned no usable price or availability for {url} (raw: {raw!r})")
            return None

        logger.info(
            f"LLM restock fallback result: price={price} currency={currency} "
            f"availability={availability!r} url={url}"
        )
        return {
            'price': price,
            'currency': currency,
            'availability': availability,
            '_tokens': tokens,
            '_input_tokens': input_tokens,
            '_output_tokens': output_tokens,
            '_model': llm_cfg['model'],
        }

    except json.JSONDecodeError as e:
        logger.warning(f"LLM restock fallback: JSON parse failed ({e}) - raw response was: {raw!r}")
        return None
    except Exception as e:
        logger.warning(f"LLM restock fallback: extraction failed for {url}: {e}")
        return None
