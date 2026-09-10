# Corpus sources

This file lives outside `documents/` on purpose: ingest treats every
markdown file under that directory as corpus content, and a table of
hashes is not something the retrieval baseline should be searching.

50 Python Enhancement Proposals, fetched from the `python/peps`
repository and converted from reStructuredText to markdown by
`scripts/fetch_peps.py`. The conversion touches structure only: headings,
code blocks, admonitions and inline roles. No prose is rewritten.

PEPs are placed in the public domain or under CC0-1.0 (PEP 1), so the
corpus can be redistributed here without further condition.

The digest is the sha256 of the source `.rst` as fetched, truncated to 16
characters. It pins what was converted: upstream may edit a PEP, and this
is how a later re-run is known to have produced a different corpus.

| PEP | Title | Source | sha256 |
| --- | --- | --- | --- |
| 8 | Style Guide for Python Code | [pep-0008.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0008.rst) | `6028935c6cb2c674` |
| 20 | The Zen of Python | [pep-0020.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0020.rst) | `742999637cc96eef` |
| 202 | List Comprehensions | [pep-0202.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0202.rst) | `4ba6fabbaf7916ff` |
| 234 | Iterators | [pep-0234.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0234.rst) | `d451c27d65547374` |
| 238 | Changing the Division Operator | [pep-0238.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0238.rst) | `6f77649f5a45d71b` |
| 249 | Python Database API Specification v2.0 | [pep-0249.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0249.rst) | `02e3a79b4aab8d85` |
| 255 | Simple Generators | [pep-0255.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0255.rst) | `3ea93925423b575c` |
| 257 | Docstring Conventions | [pep-0257.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0257.rst) | `7db41c5aaea3bfde` |
| 263 | Defining Python Source Code Encodings | [pep-0263.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0263.rst) | `af6bd432e3c1644c` |
| 273 | Import Modules from Zip Archives | [pep-0273.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0273.rst) | `1067b7ce8c869b9c` |
| 282 | A Logging System | [pep-0282.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0282.rst) | `14f641c038c47609` |
| 289 | Generator Expressions | [pep-0289.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0289.rst) | `530c0700e00f0dd8` |
| 302 | New Import Hooks | [pep-0302.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0302.rst) | `aa1494e6b65d8a4b` |
| 308 | Conditional Expressions | [pep-0308.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0308.rst) | `8d9c84198772d43a` |
| 309 | Partial Function Application | [pep-0309.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0309.rst) | `f944b0a160b3fba4` |
| 318 | Decorators for Functions and Methods | [pep-0318.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0318.rst) | `54fc00fe48709071` |
| 328 | Imports: Multi-Line and Absolute/Relative | [pep-0328.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0328.rst) | `080222338cc51c0e` |
| 333 | Python Web Server Gateway Interface v1.0 | [pep-0333.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0333.rst) | `827ae195c08add51` |
| 342 | Coroutines via Enhanced Generators | [pep-0342.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0342.rst) | `6df96b750e9c69b0` |
| 343 | The "with" Statement | [pep-0343.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0343.rst) | `69ea4cd6143ae721` |
| 372 | Adding an ordered dictionary to collections | [pep-0372.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0372.rst) | `816d4b35af175eeb` |
| 380 | Syntax for Delegating to a Subgenerator | [pep-0380.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0380.rst) | `81b361bb5e41c3ca` |
| 393 | Flexible String Representation | [pep-0393.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0393.rst) | `f343f1ef5ac994d0` |
| 405 | Python Virtual Environments | [pep-0405.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0405.rst) | `97a8048cbf102953` |
| 420 | Implicit Namespace Packages | [pep-0420.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0420.rst) | `204c149993e26ec6` |
| 428 | The pathlib module -- object-oriented filesystem paths | [pep-0428.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0428.rst) | `b6f9af6912b1bab8` |
| 435 | Adding an Enum type to the Python standard library | [pep-0435.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0435.rst) | `4ad6df5b0699dcb5` |
| 440 | Version Identification and Dependency Specification | [pep-0440.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0440.rst) | `73e66234348131cc` |
| 443 | Single-dispatch generic functions | [pep-0443.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0443.rst) | `44897a2c82e72507` |
| 448 | Additional Unpacking Generalizations | [pep-0448.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0448.rst) | `a863f3401556d392` |
| 465 | A dedicated infix operator for matrix multiplication | [pep-0465.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0465.rst) | `d4ed1e3f7a658bfd` |
| 484 | Type Hints | [pep-0484.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0484.rst) | `ddfe61c36a61b3ba` |
| 492 | Coroutines with async and await syntax | [pep-0492.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0492.rst) | `8e60fb391f3584cb` |
| 498 | Literal String Interpolation | [pep-0498.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0498.rst) | `a3eb06e347f3bb39` |
| 503 | Simple Repository API | [pep-0503.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0503.rst) | `375c6ad162214646` |
| 508 | Dependency specification for Python Software Packages | [pep-0508.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0508.rst) | `493a519a280e13b8` |
| 515 | Underscores in Numeric Literals | [pep-0515.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0515.rst) | `87c8c0e0936626ec` |
| 517 | A build-system independent format for source trees | [pep-0517.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0517.rst) | `fcc588817c7d8223` |
| 518 | Specifying Minimum Build System Requirements for Python Projects | [pep-0518.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0518.rst) | `511eef59d353f152` |
| 525 | Asynchronous Generators | [pep-0525.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0525.rst) | `ae680b2c3e498242` |
| 544 | Protocols: Structural subtyping (static duck typing) | [pep-0544.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0544.rst) | `d02ac1f85da2b533` |
| 557 | Data Classes | [pep-0557.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0557.rst) | `5f616e7d2fb9a563` |
| 561 | Distributing and Packaging Type Information | [pep-0561.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0561.rst) | `0166f88fb4de2eaa` |
| 563 | Postponed Evaluation of Annotations | [pep-0563.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0563.rst) | `5512e8a43ace196d` |
| 567 | Context Variables | [pep-0567.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0567.rst) | `d3ed88ae2ec34aa2` |
| 572 | Assignment Expressions | [pep-0572.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0572.rst) | `abf5401ec523a03d` |
| 585 | Type Hinting Generics In Standard Collections | [pep-0585.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0585.rst) | `918bf996d429379f` |
| 604 | Allow writing union types as `X | Y` | [pep-0604.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0604.rst) | `c6d87a6c7ea65964` |
| 634 | Structural Pattern Matching: Specification | [pep-0634.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0634.rst) | `352927baffdbb0fd` |
| 695 | Type Parameter Syntax | [pep-0695.rst](https://raw.githubusercontent.com/python/peps/main/peps/pep-0695.rst) | `115c754fb36b634b` |
