# Machine Translation and Similarity Search with Locality-Sensitive Hashing (LSH)

## Project Overview

This project demonstrates advanced natural language processing techniques combining cross-lingual word embedding alignment and approximate nearest neighbor search using Locality-Sensitive Hashing (LSH). The implementation focuses on transforming and comparing word and document embeddings across different languages and finding similar documents efficiently.

## Key Features

- **Cross-Lingual Word Embedding Alignment**
  - Map word embeddings between English and French vector spaces
  - Use gradient descent to learn optimal transformation matrix
  - Compute alignment accuracy

- **Document Embedding Generation**
  - Create document vectors by aggregating word embeddings
  - Process and transform text documents (tweets in this example)

- **Approximate Nearest Neighbor Search**
  - Implement Locality-Sensitive Hashing (LSH) for fast similarity search
  - Reduce computational complexity of finding similar documents
  - Support multi-universe hashing for improved search accuracy

## Technologies and Libraries

- Python
- NumPy
- NLTK
- Pickle
- Word Embeddings
- Locality-Sensitive Hashing

## Main Components

1. **Word Embedding Alignment**
   - Gradient descent optimization
   - Loss computation
   - Transformation matrix learning

2. **Document Embedding**
   - Word embedding aggregation
   - Tweet/document vector generation

3. **Locality-Sensitive Hashing**
   - Random projection planes
   - Hash table creation
   - Approximate k-nearest neighbors search

## Installation

```bash
# Clone the repository
git clone https://github.com/Ayoub-Elkhaiari/Machine_Translation_LSH.git

# Install required dependencies
pip install numpy nltk
```

## Usage

```python
# Example of finding similar documents
doc_id = 0
nearest_neighbor_ids = approximate_knn(
    doc_id, 
    vec_to_search, 
    planes_l, 
    hash_tables, 
    id_tables, 
    k=3, 
    num_universes_to_use=5
)
```

## Performance Considerations

- LSH provides faster approximate nearest neighbor search
- Trade-off between search speed and exact matching
- Configurable parameters like number of planes and universes

## Potential Improvements

- Extend to more languages
- Implement more advanced embedding techniques
- Enhance LSH accuracy and performance
- Add visualization of embedding spaces
- Adding Attention mechanism

## References

- Word2Vec
- Locality-Sensitive Hashing
- Cross-Lingual Embedding Alignment

## License

MIT LICENCE

## Contributing

Contributions, issues, and feature requests are welcome!

---

*Developed as an exploration of machine learning techniques in natural language processing*
