from sri.trie import Trie
import json


def main() -> None:
    trie = Trie()
    words = ["hola", "mundo", "python", "programación"]

    # Insertar palabras en el Trie
    for word in words:
        trie.insert(word)

    # Cargar el Trie desde un archivo JSON
    with open('trie.json', 'r') as f:
        trie.from_json(f.read())

    print(trie.find_closest_words('p',2))


if __name__ == "__main__":
    main()
