# RFxpl
Random Forests eXplainer with SAT


### Usage:


* Print the Usage:

<code>$> ./RFxp.py -h </code>

* Train an RF model:

<code>$> ./RFxp.py -d 4 -n 100  -t  ./tests/iris/iris.csv </code>

* Compute an explanation:

<code>$> ./RFxp.py -X abd -x '5.4,3.0,4.5,1.5' ./tests/iris/iris_nbestim_100_maxdepth_6.mod.pkl </code>

<code>$> ./RFxp.py -X con -x '5.4,3.0,4.5,1.5' ./tests/iris/iris_nbestim_100_maxdepth_6.mod.pkl </code>


* Increase verbosity: 
Print the explanation feature-values by increasing the verbosity.

<code>$> ./RFxp.py -v -X abd -x '5.4,3.0,4.5,1.5' ./tests/iris/iris_nbestim_100_maxdepth_6.mod.pkl .tests/iris/iris.csv </code>

* Inflate an explanation:
Compute inflated explanation when data features are categorical or/and ordinal.

<code>$> ./infxp/Infxpl.py -v  -X abd -x '0,0,0,2,0,1,4,0,2,2,1,3' -c ./infxp/tests/adult/adult_nbestim_100_maxdepth_8.mod.pkl ./infxp/tests/adult/adult.csv </code>


## Citations

Please cite the following paper when you use this work:

```
@inproceedings{ims-ijcai21,
  author       = {Yacine Izza and
                  Jo{\~{a}}o Marques{-}Silva},
  editor       = {Zhi{-}Hua Zhou},
  title        = {On Explaining Random Forests with {SAT}},
  booktitle    = {Proceedings of the Thirtieth International Joint Conference on Artificial
                  Intelligence, {IJCAI} 2021, Virtual Event / Montreal, Canada, 19-27
                  August 2021},
  pages        = {2584--2591},
  publisher    = {ijcai.org},
  year         = {2021},
  url          = {https://doi.org/10.24963/ijcai.2021/356},
  doi          = {10.24963/ijcai.2021/356}
}

@article{iism-aaai24,
  author       = {Yacine Izza and
                  Alexey Ignatiev and
                  Peter J. Stuckey and
                  Jo{\~{a}}o Marques{-}Silva},
  title        = {Delivering Inflated Explanations},
  journal      = {CoRR},
  volume       = {abs/2306.15272},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2306.15272},
  doi          = {10.48550/ARXIV.2306.15272},
  eprinttype    = {arXiv},
  eprint       = {2306.15272},
  timestamp    = {Fri, 30 Jun 2023 15:53:15 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2306-15272.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.