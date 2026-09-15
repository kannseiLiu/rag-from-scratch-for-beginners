# Original learning draft

`original_pdf_rag.py` preserves the user's first PDF RAG learning draft. Its
only behavior change is that the PDF is supplied as the first command-line
argument:

```sh
python legacy/original_pdf_rag.py path/to/document.pdf
```

One pre-existing trailing space was normalized so repository whitespace checks
pass; it does not affect the draft's behavior.

For the maintained, tested workflow, use the `pdf-rag` command and the
progressive examples in `examples/` instead.
