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

import math
from collections import defaultdict
import mmh3


class BM25:
    def __init__(self, documents):
        self.documents = documents
        self.N = len(documents)
        self.avgdl = 0.0
        self.idf = {}
        self.doc_freq = defaultdict(int)
        self.calc_idf()
        self.calc_avgdl()

    def calc_idf(self):
        for doc in self.documents:
            for word in set(doc):
                self.doc_freq[word] += 1
        for word, freq in self.doc_freq.items():
            self.idf[word] = math.log((self.N - freq + 0.5) / (freq + 0.5)) + 1

    def calc_avgdl(self):
        total_dl = 0
        for doc in self.documents:
            total_dl += len(doc)
        self.avgdl = total_dl / self.N if self.N > 0 else 0

    def bm25_score(self, query, doc_idx, k1=1.5, b=0.75):
        score = 0.0
        doc = self.documents[doc_idx]
        dl = len(doc)
        for word in query:
            if word not in self.idf:
                continue
            idf = self.idf[word]
            tf = doc.count(word)
            numerator = idf * tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / self.avgdl)
            score += numerator / denominator
        return score

    def get_scores(self, query, k1=1.5, b=0.75):
        scores = {}
        for doc_idx in range(self.N):
            score = self.bm25_score(query, doc_idx, k1, b)
            scores[doc_idx] = score
        return scores


class MinHash:
    def __init__(self, num_perm=128):
        self.num_perm = num_perm

    def _hash(self, content, seed):
        return mmh3.hash(content, seed)

    def text_to_minhash(self, text:str):
        words = text.split()
        minhash = [float('inf')] * self.num_perm

        for word in words:
            for i in range(self.num_perm):
                hash_value = self._hash(word, i)
                if hash_value < minhash[i]:
                    minhash[i] = hash_value

        return minhash

    def similarity(self, text1, text2):
        minhash1 = self.text_to_minhash(text1)
        minhash2 = self.text_to_minhash(text2)

        match = 0
        for h1, h2 in zip(minhash1, minhash2):
            if h1 == h2:
                match += 1

        return match / self.num_perm

if __name__ == "__main__":
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog leaps over a lazy fox",
        "The quick brown fox jumps over a lazy dog",
        "Hello world, this is a test sentence"
    ]

    minhash = MinHash(num_perm=128)

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity_score = minhash.similarity(texts[i], texts[j])
            print(f"Similarity between text {i} and text {j}: {similarity_score:.4f}")

    snippets = [
        ["the", "quick", "brown", "fox"],
        ["jumps", "over", "the", "lazy", "dog"],
        ["the", "brown", "dog"],
        ["quick", "fox", "jumps"]
    ]
    bm25 = BM25(snippets)
    query = ["quick", "brown", "fox"]
    scores = bm25.get_scores(query)

    print("Query:", query)
    print("BM25 Scores for each snippet:")
    for doc_idx, score in scores.items():
        print(f"Snippet {doc_idx}: {score:.4f}")
    