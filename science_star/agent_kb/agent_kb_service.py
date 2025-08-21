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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from agent_kb_retrieval import AKB_Manager
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import time
import os

MAX_CONCURRENT_SEARCHES = int(os.getenv("MAX_CONCURRENT_SEARCHES", 10))
CACHE_TTL = 60

app = FastAPI(title="Optimized Knowledge Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = AKB_Manager(json_file_paths=["./agent_kb/agent_kb_database.json"])

performance_stats = {
    "total_requests": 0,
    "avg_response_time": 0.0,
    "last_updated": time.time()
}

response_cache = {}

class SearchRequest(BaseModel):
    query: str
    top_k: int = 1
    weights: Optional[Dict[str, float]] = {'text': 0.5, 'semantic': 0.5}

class WorkflowResponse(BaseModel):
    workflow_id: str
    total_score: Optional[float]
    query: str
    plan: Optional[str] = None
    search_plan: Optional[str] = None
    agent_experience: Optional[str]
    search_agent_experience: Optional[str]

class PerformanceStats(BaseModel):
    total_requests: int
    avg_response_time: float
    cache_hit_rate: float

def update_performance_stats(response_time: float):
    total_time = performance_stats["avg_response_time"] * performance_stats["total_requests"]
    performance_stats["total_requests"] += 1
    performance_stats["avg_response_time"] = (total_time + response_time) / performance_stats["total_requests"]
    performance_stats["last_updated"] = time.time()

@app.post("/search/hybrid", response_model=List[WorkflowResponse])
async def hybrid_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"hybrid_{request.query}_{request.top_k}"
    
    try:
        if cache_key in response_cache:
            if time.time() - response_cache[cache_key]["timestamp"] < CACHE_TTL:
                return response_cache[cache_key]["data"]
        
        results = manager.hybrid_search(
            query=request.query,
            top_k=request.top_k,
            weights=request.weights
        )
        
        response_data = [WorkflowResponse(
            workflow_id=item['workflow_id'],
            total_score=item['total_score'],
            query=item['query'],
            plan=item['plan'],
            search_plan = item['search_plan'],
            agent_experience=item['agent_experience'],
            search_agent_experience=item['search_agent_experience']
        ) for item in results]
        
        response_cache[cache_key] = {
            "timestamp": time.time(),
            "data": response_data
        }
        
        update_performance_stats(time.time() - start_time)
        
        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/text", response_model=List[WorkflowResponse])
async def text_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"text_{request.query}_{request.top_k}"
    
    try:
        if cache_key in response_cache:
            if time.time() - response_cache[cache_key]["timestamp"] < CACHE_TTL:
                return response_cache[cache_key]["data"]
        
        raw_results = manager.search_by_text(request.query, "query", request.top_k)
        
        response_data = [WorkflowResponse(
            workflow_id=item['workflow_id'],
            total_score=item['score'],
            query=item['content']['query'],
            plan=item['content']['plan'],
            search_plan = item['content']['search_plan'],
            agent_experience=item['content']['agent_experience'],
            search_agent_experience=item['content']['search_agent_experience']
        ) for item in raw_results]
        
        response_cache[cache_key] = {
            "timestamp": time.time(),
            "data": response_data
        }
        
        update_performance_stats(time.time() - start_time)
        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text search failed: {str(e)}")

@app.post("/search/semantic", response_model=List[WorkflowResponse])
async def semantic_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"semantic_{request.query}_{request.top_k}"
    
    try:
        if cache_key in response_cache:
            if time.time() - response_cache[cache_key]["timestamp"] < CACHE_TTL:
                return response_cache[cache_key]["data"]
        
        raw_results = manager.search_by_semantic(request.query, "query", request.top_k)
        
        response_data = [WorkflowResponse(
            workflow_id=item['workflow_id'],
            total_score=item['score'],
            query=item['content']['query'],
            plan=item['content']['plan'],
            search_plan = item['content']['search_plan'],
            agent_experience=item['content']['agent_experience'],
            search_agent_experience=item['content']['search_agent_experience']
        ) for item in raw_results]
        
        response_cache[cache_key] = {
            "timestamp": time.time(),
            "data": response_data
        }
        
        update_performance_stats(time.time() - start_time)
        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

@app.get("/performance", response_model=PerformanceStats)
async def get_performance():
    cache_hit_rate = sum(1 for v in response_cache.values() if time.time() - v["timestamp"] < CACHE_TTL) / len(response_cache) if response_cache else 0
    
    return {
        "total_requests": performance_stats["total_requests"],
        "avg_response_time": performance_stats["avg_response_time"],
        "cache_hit_rate": cache_hit_rate
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=int(os.getenv("UVICORN_WORKERS", 1)),
        limit_concurrency=MAX_CONCURRENT_SEARCHES
    )