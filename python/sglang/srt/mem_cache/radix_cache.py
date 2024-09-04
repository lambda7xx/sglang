from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, List, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = time.time()

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


class RadixCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: BaseTokenToKVPool,
        disable: bool = False,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.disable = disable
        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0

    def match_prefix(self, key: List, **kwargs):
        print(f"1 python/sglang/srt/mem_cache/radix_cache.py match_prefix: key:{key} and kwargs:{kwargs} and self.disable:{self.disable}")
        if self.disable:
            return [], self.root_node

        value = []
        last_node = [self.root_node]
        self._match_prefix_helper(self.root_node, key, value, last_node) #xiao:0823 这个很重要
        print(f"1.5 python/sglang/srt/mem_cache/radix_cache.py match_prefix: value:{value} and last_node:{last_node[0]}")
        if value:
            value = torch.concat(value)
        else:
            value = torch.tensor([], dtype=torch.int32)
        print(f"2 python/sglang/srt/mem_cache/radix_cache.py match_prefix: value.shape:{value.shape} and last_node:{last_node[0]}")
        return value, last_node[0]

    def insert(self, key: List, value=None):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)

    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it finishes."""
        if token_ids is None:
            token_ids = (req.origin_input_ids + req.output_ids)[:-1] #xiao: 除去最后一个，取这个list的前面所有的元素
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ] #xiao: 0904 ： w为什么这么取？，req.req_pool_idx是什么？这个是什么意思？
        print(f"1 python/sglang/srt/mem_cache/radix_cache.py RadixCache::cache_finished_req: len(token_ids):{len(token_ids)} ")
        print(f"2 python/sglang/srt/mem_cache/radix_cache.py RadixCache::cache_finished_req: type(kv_indices):{type(kv_indices)} and type(req.req_pool_index):{type(req.req_pool_idx)} ")
        if self.disable:
            self.token_to_kv_pool.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(token_ids, kv_indices.clone()) #xiao: new_prefix_len是表示当前这个req和radix_tree cache的最长的prefix的长度
        self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices) : new_prefix_len]) #xiao:0904 为什么free?

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx) #0904 为什么free?
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        if token_ids is None:
            token_ids = req.fill_ids

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(token_ids, kv_indices.clone())
        print(f"1 python/sglang/srt/mem_cache/radix_cache.py RadixCache::cache_unfinished_req, new_prefix_len:{new_prefix_len}")
        self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices) : new_prefix_len])

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node = self.match_prefix(token_ids) #xiao 0904 为什么这样做

        assert len(new_indices) == len(token_ids) #xiao 0904 :为什么len(new_indices) == len(token_ids)
        self.req_to_token_pool.req_to_token[
            req.req_pool_idx, len(req.prefix_indices) : len(new_indices)
        ] = new_indices[len(req.prefix_indices) :] #xiao 0904 为什么这样做

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
        req.prefix_indices = new_indices
        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper(self.root_node)

    def evict(self, num_tokens: int, evict_callback: Callable):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            evict_callback(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    #xiao 0823 这个很重要,干什么的
    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:#当lock_ref为1时，然后就在以后被evict掉
                self.evictable_size_ += len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_ #xiao: evict掉大小

    ##### Internal Helper Functions #####
    #xiao 0823 这个很重要
    def _match_prefix_helper(
        self, node: TreeNode, key: List, value, last_node: TreeNode
    ):
        node.last_access_time = time.time()
        if len(key) == 0:
            return
        print(f"1 python/sglang/srt/mem_cache/radix_cache.py RadixCache::_match_prefix_helper: len(key):{len(key)} and key:{key} and key[0]:{key[0]}")
        print(f"2 python/sglang/srt/mem_cache/radix_cache.py RadixCache::_match_prefix_helper: node.children.keys():{node.children.keys()}")
        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key) #xiao:0823 这个很重要
            print(f"3 python/sglang/srt/mem_cache/radix_cache.py RadixCache::_match_prefix_helper: prefix_len:{prefix_len} and child.key:{len(child.key)} ")
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                last_node[0] = new_node
            else:
                value.append(child.value)
                last_node[0] = child
                self._match_prefix_helper(child, key[prefix_len:], value, last_node)

    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len:][0]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[key[:split_len][0]] = new_node
        return new_node

    #xiao 0823 这个很重要
    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.time()
        if len(key) == 0:
            return 0

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key) #xiao: 求child.key和key的最长公共前缀

            if prefix_len == len(child.key):
                if prefix_len == len(key):
                    return prefix_len
                else:
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, value)

            new_node = self._split_node(child.key, child, prefix_len)
            return prefix_len + self._insert_helper(
                new_node, key[prefix_len:], value[prefix_len:]
            )

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[key[0]] = new_node
            self.evictable_size_ += len(value)
        return 0

    def _print_helper(self, node: TreeNode, indent: int):
        for _, child in node.children.items():
            print(" " * indent, len(child.key), child.key[:10], f"r={child.lock_ref}")
            self._print_helper(child, indent=indent + 2)

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self, node: TreeNode):
        x = len(node.value)
        for child in node.children.values():
            x += self._total_size_helper(child)
        return x

    def _collect_leaves(self):
        ret_list = []

        def dfs_(cur_node):
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)

            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list


if __name__ == "__main__":
    tree = RadixCache(None, None, False)

    tree.insert("Hello")
    tree.insert("Hello")
    tree.insert("Hello_L.A.!")
    # tree.insert("Hello_world! Happy")
    # tree.insert("I love you!")
    tree.pretty_print()

    # print(tree.match_prefix("I love you! aha"))

    # def evict_callback(x):
    #    print("evict", x)
    #    return len(x)

    # tree.evict(5, evict_callback)
    # tree.evict(10, evict_callback)
    # tree.pretty_print()
