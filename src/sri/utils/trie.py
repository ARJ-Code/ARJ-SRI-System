import json
from typing import Dict, Optional, List


class TrieNode:
    def __init__(self, char: str = '~', is_end_of_word: bool = False) -> None:
        """
        Initializes a TrieNode.

        Args:
            char (str): The character associated with this node.
            is_end_of_word (bool): Indicates whether this node represents the end of a word.
        """
        self.char: str = char
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = is_end_of_word


class Trie:
    def __init__(self) -> None:
        """
        Initializes a Trie data structure.
        """
        self.root: TrieNode = TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the Trie.

        Args:
            word (str): The word to insert.
        """
        node: TrieNode = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode(char)
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """
        Searches for a word in the Trie.

        Args:
            word (str): The word to search for.

        Returns:
            bool: True if the word exists in the Trie, False otherwise.
        """
        node: TrieNode = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """
        Checks if any word in the Trie starts with the given prefix.

        Args:
            prefix (str): The prefix to check.

        Returns:
            bool: True if there is a word with the given prefix, False otherwise.
        """
        node: TrieNode = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def find_closest_words(self, prefix: str, k: int) -> List[str]:
        """
        Finds the closest k words to the given prefix in the Trie.

        Args:
            prefix (str): The prefix to search for.
            k (int): The maximum number of closest words to return.

        Returns:
            List[str]: A list of closest words.
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        def dfs(node: TrieNode, word: str, result: List[str]) -> None:
            if len(result) == k:
                return
            if node.is_end_of_word:
                result.append(word)
            for char in node.children:
                dfs(node.children[char], word + char, result)

        result: List[str] = []
        dfs(node, prefix, result)
        return result

    def to_json(self, node: Optional[TrieNode] = None) -> Dict[str, str] | str:
        """
        Converts the Trie to a JSON representation.

        Args:
            node (Optional[TrieNode]): The starting node for conversion.

        Returns:
            Dict[str, str] | str: The JSON representation of the Trie.
        """
        if node is None:
            node = self.root
        if node.is_end_of_word:
            return '$'
        children = {char: self.to_json(child)
                    for char, child in node.children.items()}
        return children

    def from_json(self, json_str: str) -> None:
        self.root = self._from_json(json.loads(json_str))

    def _from_json(self, node: Dict[str, str] | str) -> TrieNode:
        if isinstance(node, str):
            return TrieNode(char=node[-1], is_end_of_word=True)
        else:
            trie_node = TrieNode()
            for char, child in node.items():
                trie_node.children[char] = self._from_json(child)
            return trie_node
