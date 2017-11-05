# Triple_Classifier

This is the Github for my research under Daniel Bauer, Lecturer in CS at Columbia University

Program/Method designed to classify a triple (subject-relation-predicate) in an [AMR](https://amr.isi.edu/) graph as either good or bad based on how frequently similar triples appeared in other sentences (each sentence is represented by an AMR). 'Similar' is defined as triples whose components have similar vectors in a word embedding. This project attempts to use feed forward neural networks to do the classification.
