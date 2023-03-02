# csi5180-a2
This repository contains our implementation for Assignment 2 of CSI 5180 AI: Virtual Assistants.


To replicate our results:

```
python -m paraphrase --mode train --alg bert
```

This command would run a certain classifier on a certain dataset. The evaluation scores will be outputed in the end.

--mode should be from [train, dev, test]

--alg should be from [em, wer, bert]

em = exact match classifier

wer = word edit distance classifier

bert = BERT-based text similarity classifier
