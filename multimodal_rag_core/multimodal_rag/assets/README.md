# Bundled synonym map

`synonyms.json.gz` is a compact, offline query-expansion resource. It is loaded
only when `KEYWORD_EXPANSION` is `SYNONYMS` or `CORPUS+SYNONYMS`.

- Source: WordNet 3.0 (Princeton University), using only synset lemmas.
- Root filter: Moby Words II `common.txt` (public domain, Grady Ward).
- Rules: lowercase single alphabetic words, 3–24 characters; stopwords removed;
  proper names and multi-word entries excluded; at most 10 synonyms per root.
- Ranking: synonyms with SemCor lemma counts first, then by the number of
  synsets in which they co-occur with the root, then alphabetically.
- Runtime: standard library only (`gzip` + `json`); no NLTK, no WordNet
  installation, and no network access.

The generated file is about 327 KB gzipped. See `WORDNET_LICENSE.txt` for the
WordNet 3.0 license. Moby Words II is public-domain material by Grady Ward.

To regenerate, use an offline development environment with WordNet 3.0 and
Moby `common.txt`; the bundled file is the shipped artifact, not a runtime
dependency.
