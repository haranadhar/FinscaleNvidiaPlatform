#!/usr/bin/env python3
"""Download SEC 10-K filing text via the free SEC submissions/Archives endpoints."""
import argparse, gzip, html, json, os, re, sys, time, zlib
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Default cross-sector universe: exactly 30 distinct public companies.
DEFAULT_TICKERS = ["AAPL","NVDA","MSFT","AMZN","GOOGL","META","TSLA","JPM","BAC","WFC","GS","MS","V","MA","PYPL","NFLX","DIS","WMT","COST","HD","UNH","JNJ","PFE","ABBV","XOM","CVX","BA","CAT","AMD","INTC"]

TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
class TextExtractor(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[]; self.skip=0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "ix:header"): self.skip += 1
        if tag in ("p","div","br","tr","section","h1","h2","h3"): self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in ("script", "style", "ix:header") and self.skip: self.skip -= 1
        if tag in ("p","div","tr","section","h1","h2","h3"): self.parts.append("\n")
    def handle_data(self, data):
        if not self.skip: self.parts.append(data)
def fetch(url, ua):
    req=Request(url, headers={"User-Agent":ua,"Accept-Encoding":"gzip, deflate"})
    with urlopen(req, timeout=60) as r:
        body=r.read()
        encoding=r.info().get("Content-Encoding", "").lower()
        if encoding=="gzip": return gzip.decompress(body)
        if encoding=="deflate":
            try: return zlib.decompress(body)
            except zlib.error: return zlib.decompress(body, -zlib.MAX_WBITS)
        return body
def strip_html(raw):
    parser=TextExtractor(); parser.feed(raw.decode("utf-8",errors="replace"))
    text=html.unescape(" ".join(parser.parts)); text=re.sub(r"[ \t\xa0]+"," ",text)
    text=re.sub(r" *\n *","\n",text); return re.sub(r"\n{3,}","\n\n",text).strip()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers",nargs="+",default=None); ap.add_argument("--tickers-file",type=Path,default=None,help="Optional text file, one ticker per line"); ap.add_argument("--output",type=Path,default=Path("data/raw/edgar_10k.jsonl"))
    ap.add_argument("--limit",type=int,default=1); ap.add_argument("--pause",type=float,default=.25)
    a=ap.parse_args(); ua=os.environ.get("SEC_USER_AGENT","").strip()
    if a.limit < 1 or a.pause < 0.1: ap.error("--limit must be >=1 and --pause >=0.1 seconds")
    if a.tickers_file and a.tickers: ap.error("Choose either --tickers or --tickers-file")
    if a.tickers_file:
        tickers=[line.strip().upper() for line in a.tickers_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    else: tickers=a.tickers or DEFAULT_TICKERS
    tickers=list(dict.fromkeys(t.upper() for t in tickers))
    if not tickers: ap.error("No tickers were provided")
    print(f"Requested {len(tickers)} companies: {', '.join(tickers)}")
    if not ua or "@" not in ua: ap.error("Set SEC_USER_AGENT to a truthful contact identity, e.g. 'Research Name contact@example.org'; SEC requires a declared User-Agent")
    try: ticker_directory=json.loads(fetch(TICKER_URL,ua))
    except (HTTPError,URLError,TimeoutError) as e: sys.exit(f"SEC ticker lookup failed: {e}")
    lookup={str(v["ticker"]).upper():str(v["cik_str"]).zfill(10) for v in ticker_directory.values()}
    a.output.parent.mkdir(parents=True,exist_ok=True); count=0
    with a.output.open("w",encoding="utf-8") as out:
        for ticker in tickers:
            symbol=ticker.upper()
            if symbol not in lookup: print(f"Unknown ticker: {symbol}",file=sys.stderr); continue
            cik=lookup[symbol]
            try:
                recent=json.loads(fetch(SUBMISSIONS_URL.format(cik=cik),ua))["filings"]["recent"]
                chosen=[i for i,f in enumerate(recent["form"]) if f=="10-K"][:a.limit]
                for i in chosen:
                    accession=recent["accessionNumber"][i]; filed=recent["filingDate"][i]; doc=recent["primaryDocument"][i]
                    url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{doc}"
                    time.sleep(a.pause); text=strip_html(fetch(url,ua))
                    if text:
                        row={"ticker":symbol,"cik":cik,"form":"10-K","filing_date":filed,"accession":accession,"source_url":url,"text":text}
                        out.write(json.dumps(row,ensure_ascii=False)+"\n"); out.flush(); count+=1
                        print(f"Fetched {symbol} {filed} ({len(text):,} chars)")
                    time.sleep(a.pause)
            except (HTTPError,URLError,TimeoutError,KeyError,json.JSONDecodeError) as e: print(f"Failed {symbol}: {e}",file=sys.stderr)
    print(f"Wrote {count} filing(s) across the requested {len(tickers)} companies: {a.output}")
    if count < len(tickers): print("Some companies may have missing filings; review stderr and the output JSONL.", file=sys.stderr)
if __name__=="__main__": main()
