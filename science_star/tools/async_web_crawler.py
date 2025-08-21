#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. Personal AI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mimetypes
import os
import pathlib
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urljoin, urlparse

import requests
from serpapi import GoogleSearch, BingSearch

from smolagents import Tool, OpenAIServerModel

from .cookies import COOKIES

import asyncio
from crawl4ai import AsyncWebCrawler

from rag.embeddings import OpenAIEmbedding
from rag.retrievers import SimpleVectorRetriever

from .similarity import MinHash

import requests
import yaml
from reflectors import SearchReflector


class SimpleCrawler:
    '''
        @overview: a simple crawler for agent to search contents related to query, crawl pages through url.etc
        @func:

    '''
    def __init__(self,
                 serpapi_key: Optional[str] = None,
                 model: OpenAIServerModel=None,
                 reflection: bool=True,
                 roll_out: int=0,
                 search_limit: int=10,
                 serp_num: int=10,
                 topk: int=1,
                 rerank: bool=False,
                 chunk_size: int=500,
                 chunk_overlap: int=50,
                 use_db: Optional[bool] = False,
                 path: Optional[str] = None,
                 ):
        self.serpapi_key = os.getenv("SERP_API_KEY") if serpapi_key is None else serpapi_key
        self.model = model
        if model is not None and reflection:
            self.reflector = SearchReflector(model=model)
            self.reflection = True
        else:
            self.reflection = False
        self.history = []
        self.roll_out = roll_out
        self.serp_num = serp_num
        self.search_limit = search_limit
        self.topk = topk
        self.rerank = rerank
        
        self.retriever = SimpleVectorRetriever(embedding_model=OpenAIEmbedding(),
                                               chunk_size=chunk_size, 
                                               chunk_overlap=chunk_overlap,
                                               path=None if not use_db else path)
    
    def _search(self, query: str, filter_year: Optional[int] = None) -> List[str]:
        if self.serpapi_key is None:
            raise ValueError("Missing SerpAPI key.")
        
        self.history.append((query, time.time()))

        params = {
            "engine": "google", # google
            "q": query,
            "api_key": self.serpapi_key,
            "num": self.serp_num
        }
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        search = GoogleSearch(params)

        results = search.get_dict()
        '''
        @ serp result format -> json dict
        dict_keys(['search_metadata', 
                    'search_parameters', 
                    'search_information', 
                    'knowledge_graph', 
                    'inline_images', 
                    'related_questions', 
                    'organic_results', 
                    'top_stories_link', 
                    'top_stories_serpapi_link', 
                    'related_searches', 
                    'pagination', 
                    'serpapi_pagination']
                    )
        '''

        self.page_title = f"{query} - Search"
        if "organic_results" not in results.keys():
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        if len(results["organic_results"]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            return f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."

        web_snippets: List[str] = list()
        idx = 0
        if "organic_results" in results:
            for page in results["organic_results"]:
                idx += 1
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                _search_result = {
                    "idx": idx,
                    "title": page["title"],
                    "date": date_published,
                    "snippet": snippet,
                    "source": source,
                    "link": page['link']
                }
                
                web_snippets.append(_search_result)

        return web_snippets

    
    def _pre_visit(self, url):
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i][0] == url:
                return f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
        return ""


    def _to_contents(self, query:str, snippets:List):
        web_snippets = []
        idx=1
        for search_info in snippets:
            redacted_version = f"{idx}. [{search_info['title']}]({search_info['link']})" + \
                            f"{search_info['date']}{search_info['source']}\n{self._pre_visit(search_info['link'])}{search_info['snippet']}"
            redacted_version = redacted_version.replace("Your browser can't play this video.", "")
            web_snippets.append(redacted_version)
            idx+=1
        
        content = (
            f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
            + "\n\n".join(web_snippets)
        )
        return content
    
    def _expand_query(self, query:str) -> List[str]:
        prompted_query = self.query_rollout_prompt_template.format(query=query, roll_out=self.roll_out)
        input_messages = [
            {
                "role": "user",
                "content": prompted_query,
            }
        ]
        chat_message = self.model(
            messages = input_messages,
            stop_sequences=["<end>"],
        )
        model_output = chat_message.content
        try:
            queries = model_output.split('<begin>')[1].strip()
            queries = queries.split("\n")[:self.roll_out]
        except:
            queries = []
        queries.append(query)

        return queries
    
    async def _crawl_page(self, url):
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(url=url)
            return result.markdown
        
    def _read_page(self, url):
        jina_url = f'https://r.jina.ai/{url}'
        headers = {
            'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}',
            'X-Engine': 'direct',
            'X-Return-Format': 'text',
            'X-Timeout': '10',
            'X-Token-Budget': '50000'
        }
        response = requests.get(jina_url, headers=headers)
        return response.text

    
    def _query(self, query, contents):
        return self.retriever.retrieve(query=query, contents=contents)
    
    def _check_history(self, url_or_query):
        header=''
        for i in range(len(self.history) - 2, -1, -1):  # Start from the second last
            if self.history[i][0] == url_or_query:
                header += f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
                return header
        self.history.append((url_or_query, time.time()))
        return header
    
    
    def evaluate_similarity_score(self, ref_snippets, info, style="minhash"):
        snippet = info['snippet']
        title = info['title']
        idx = info['idx']
        score = 0

        assert style in ["minhash", 'bm25', 'llm_score']

        if style=="minhash":
            minhash = MinHash(num_perm=128)
            for ref_snippet in ref_snippets:
                ref = ref_snippet['title'] + ref_snippet['snippet']
                score += minhash.similarity(ref, title+snippet)
            return score / len(ref_snippets)
        
        elif style=="llm_score":
            prompt = self.evaluate_prompt_template.format(query="query", idx=idx, title=title, snippet=snippet)
            input_messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            chat_message = self.model(
                messages = input_messages,
                stop_sequences=["<end>"],
            )
            model_output = chat_message.content
            score = model_output.split("score:")[1].strip()
            score = float(score)

        return score


    def aggregate(self, ref_query, search_results:Dict, intersect:bool=False, rerank:bool=False) -> List[str]:
        def _dedup(raw_list: List) -> List:
            seen_set = set()
            unique_results = []
            intersect_results = []
            for result in raw_list:
                if result['link'] in seen_set:
                    intersect_results.append(result)
                else:
                    seen_set.add(result['link'])
                    unique_results.append(result)
            return unique_results, intersect_results

        new_search_results = []
        ref_results = []
        tail_results = []
        for q, results in search_results.items():
            if q==ref_query:
                ref_results = [results[0]]+ref_results
            else:
                ref_results.append(results[0])
            tail_results += results[1:]
        new_search_results = ref_results + tail_results
        new_search_results, intersect_results = _dedup(new_search_results)

        if intersect:
            if len(intersect_results) < 2:
                print("Not enough intersect results\n")
                return search_results[ref_query]
            intersect_results, _ = _dedup(intersect_results)
            for i in range(len(intersect_results)):
                intersect_results[i]['idx'] = i+1
            return intersect_results[:self.search_limit]
        
        for i in range(len(new_search_results)):
            new_search_results[i]['idx'] = i+1

        if not rerank:
            return new_search_results[:self.search_limit]
        
        rerank_results = new_search_results[self.topk:]
        ref_results = new_search_results[:self.topk]
        
        for item in rerank_results:
            score = self.evaluate_similarity_score(ref_results, item)
            item["score"]=score
        rerank_results = sorted(rerank_results, key=lambda x: x["score"], reverse=True)
        new_search_results = ref_results + rerank_results

        for i in range(len(new_search_results)):
            new_search_results[i]['idx'] = i+1

        return new_search_results[:self.search_limit]

    def search(self, query, filter_year=None):
        use_rollout = self.roll_out > 0 and self.model is not None
        header = self._check_history(query)
        if self.reflection:
            analysis, query = self.reflector.query_reflect(query)
        if use_rollout:
            queries = self.reflector.query_rollout(query=query, n_rollout=self.roll_out)
            search_results = {}
            for q in queries:
                try:
                    snippets = self._search(q, filter_year)
                    assert len(snippets) > 0
                    search_results[q] = snippets
                except:
                    pass
            
            if len(search_results)==0:
                error_messages = f"Search for query '{query}' failed! Search query should be less specific\n"
                return error_messages
            
            web_snippets = self.aggregate(query, search_results, intersect=False, rerank=self.rerank)

        else:
            web_snippets = self._search(query, filter_year)
            if type(web_snippets)==str:
                return web_snippets
        content = self._to_contents(query, web_snippets)

        return header + content

    def crawl_page(self, url):
        header = self._check_history(url)
        pages = asyncio.run(self._crawl_page(url=url))
        return header + pages
    
    def crawl_page_with_rag(self, url, query):
        header = self._check_history(url)
        contents = self.crawl_page(url=url)
        return header + self._query(query, contents)
    
    def read_page(self, url):
        jina_url = f'https://r.jina.ai/{url}'
        headers = {
            'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}',
            'X-Engine': 'browser',
            'X-Return-Format': 'text',
            'X-Timeout': '10',
            'X-Token-Budget': '80000'
        }
        response = requests.get(jina_url, headers=headers)
        return response.text
    

class CrawlerSearchTool(Tool):
    name = "web_search"
    description = "Perform a web search query (think a google search) and returns the search results."
    inputs = {"query": {"type": "string", "description": "The web search query to perform."}}
    inputs["filter_year"] = {
        "type": "string",
        "description": "[Optional parameter]: filter the search results to only include pages from a specific year. For example, '2020' will only include pages from 2020. Make sure to use this parameter if you're trying to search for articles from a specific date!",
        "nullable": True,
    }
    output_type = "string"

    def __init__(self, 
                 crawler: SimpleCrawler,
                 rollout: int=0,
                 search_limit: int=10,
                 serp_num: int=10,
                 rerank: bool=False,
                 topk: int=1):
        super().__init__()
        # define a crawler
        self.crawler = crawler
        # reinitialize crawler's configs
        self.crawler.serp_num = serp_num
        self.crawler.roll_out = rollout
        self.crawler.search_limit = search_limit
        self.crawler.rerank = rerank
        self.crawler.topk = topk
    def forward(self, query: str, filter_year: Optional[int] = None) -> str:
        return self.crawler.search(query, filter_year)
    

class CrawlerArchiveSearchTool(Tool):
    name = "find_archived_url"
    description = "Given a url, searches the Wayback Machine and returns the archived version of the url that's closest in time to the desired date."
    inputs = {
        "url": {"type": "string", "description": "The url you need the archive for."},
        "date": {
            "type": "string",
            "description": "The date that you want to find the archive for. Give this date in the format 'YYYYMMDD', for instance '27 June 2008' is written as '20080627'.",
        },
    }
    output_type = "string"

    def __init__(self, crawler: SimpleCrawler, read_type:str="jina_read"):
        super().__init__()
        self.crawler = crawler
        self.read_type = read_type

    def forward(self, url, date) -> str:
        no_timestamp_url = f"https://archive.org/wayback/available?url={url}"
        archive_url = no_timestamp_url + f"&timestamp={date}"
        response = requests.get(archive_url).json()
        response_notimestamp = requests.get(no_timestamp_url).json()
        if "archived_snapshots" in response and "closest" in response["archived_snapshots"]:
            closest = response["archived_snapshots"]["closest"]
            print("Archive found!", closest)

        elif "archived_snapshots" in response_notimestamp and "closest" in response_notimestamp["archived_snapshots"]:
            closest = response_notimestamp["archived_snapshots"]["closest"]
            print("Archive found!", closest)
        else:
            # raise Exception(f"Your {url} was not archived on Wayback Machine, try a different url.")
            return "Your {url} was not archived on Wayback Machine, try a different url."
        target_url = closest["url"]
        
        if self.read_type == "crawl":
            content = self.crawler.crawl_page(target_url)
        else:
            content = self.crawler.read_page(target_url)

        return (
            f"Web archive for url {url}, snapshot taken at date {closest['timestamp'][:8]}:\n"
            + content
        )
    

class CrawlerReadTool(Tool):
    name = "crawl_pages"
    description = "Access a webpage using the provided URL and return completed contents of the webpage. In the case of a YouTube video URL, extract and return the video transcript."
    inputs = {
        "url": {
            "type": "string", 
            "description": "The relative or absolute url of the webpage to visit."
            },
        }
    output_type = "string"
    def __init__(self, crawler:SimpleCrawler, read_type:str="jina_read"):
        super().__init__()
        self.crawler = crawler
        self.read_type = read_type
    def forward(self, url) -> str:
        if self.read_type == "crawl":
            result = self.crawler.crawl_page(url)
        else:
            result = self.crawler.read_page(url)
        if result=='\n':
            return f"Crawling for url: {url} return None, maybe it is a url for .pdf file which is unable to crawl. " \
            "Please try to use tool: inspect_file_as_text() to get the contents."
        return result
    
    
class CrawlerRAGTool(Tool):
    name = "crawl_pages_with_retrieve"
    description = "Access a webpage using the provided URL and retrieve its text content. In the case of a YouTube video URL, extract and return the video transcript."
    inputs = {
        "url": {
            "type": "string", 
            "description": "The relative or absolute url of the webpage to visit."
            },
        "query": {
            "type": "string",
            "description": "the search query for contents relative to your task",
            },
        }
    output_type = "string"

    def __init__(self, crawler:SimpleCrawler):
        super().__init__()
        self.crawler = crawler

    def forward(self, url, query) -> str:
        return self.crawler.crawl_page_with_rag(url, query)