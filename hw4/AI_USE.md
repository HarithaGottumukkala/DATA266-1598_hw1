# AI-Use Appendix - Haritha Gottumukkala

1. Which parts did you use an assistant for, and which did you write yourself?

I used AI to help adapt the TA’s demo, add sampling methods, and
prepare the explanations and analysis. I put the notebook together,
ran the experiments, reviewed the outputs, and applied the edits.

2. Give one specific thing it produced that was wrong — a tensor shape error, a deprecated API, a loss function that trained but was wrong, a plausible-looking metric computed incorrectly. Paste the wrong output.

In an earlier assistant test, the text loader kept extra Windows
line-ending characters. It printed:

```text
Dataset: 1,796 characters; vocabulary size: 50
Training pairs: 1,668
```

3. How did you find out? What did the failure look like?

The assistant compared these counts with a normal text-file read
and identified the extra carriage-return characters. This issue
was in the earlier test, not my submitted notebook run.

4. What did you change, and why does your version work?

My notebook uses text-mode reading, which handles the line endings
consistently. The final counts are 1,738 characters, 49 unique
characters, and 1,610 pairs. I also ran the token-conversion,
output-shape, and causal-mask checks.