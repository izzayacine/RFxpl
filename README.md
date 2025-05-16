# RFxpl
Random Forests eXplainer with SAT


### Usage:


* Print the Usage:

<code>$> ./RFxp.py -h </code>

* Train an RF model:

<code>$> ./RFxp.py -d 4 -n 100  -t  ./tests/iris/iris.csv </code>

* Explain RF with majority vote (RFmv):

<code>$> ./RFxp.py -X abd -x '5.4,3.0,4.5,1.5' ./tests/iris/iris_nbestim_100_maxdepth_6.mod.pkl </code>

<code>$> ./RFxp.py -X con -x '5.4,3.0,4.5,1.5' ./tests/iris/iris_nbestim_100_maxdepth_6.mod.pkl </code>


* Explain RF with weighted vote (RFwv):

<code>$> ./RFxp.py -X abd -e mx -x '5.4,3.0,4.5,1.5' ./tests/iris/iris_nbestim_100_maxdepth_6.mod.pkl </code>

<code>$> ./RFxp.py -X con -e mx -x '5.4,3.0,4.5,1.5' ./tests/iris/iris_nbestim_100_maxdepth_6.mod.pkl </code>


* Increase verbosity: 
Print the explanation feature-values by increasing the verbosity.

<code>$> ./RFxp.py -v -X abd -x '5.4,3.0,4.5,1.5' ./tests/iris/iris_nbestim_100_maxdepth_6.mod.pkl ./tests/iris/iris.csv </code>

* Inflate an explanation:
Compute inflated explanation when data features are categorical or/and ordinal.

<code>$> ./infxp/Infxpl.py -v  -X abd -x '0,0,0,2,0,1,4,0,2,2,1,3' -c ./infxp/tests/adult/adult_nbestim_100_maxdepth_8.mod.pkl ./infxp/tests/adult/adult.csv </code>


* Most general explanation:

<code>$> ./infxp/GiAXp.py -v -X abd -M  -x '5.4,3.0,4.5,1.5' ./tests/iris/iris_nbestim_10_maxdepth_2.mod.pkl   ./tests/iris/iris.csv
</code>


## Citations

Please cite the following papers when you use this work:

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

@inproceedings{IzzaIS-aaai24,
  author       = {Yacine Izza and
                  Alexey Ignatiev and
                  Peter J. Stuckey and
                  Jo{\~{a}}o Marques{-}Silva},
  editor       = {Michael J. Wooldridge and
                  Jennifer G. Dy and
                  Sriraam Natarajan},
  title        = {Delivering Inflated Explanations},
  booktitle    = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2024, Thirty-Sixth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2024, Fourteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2014, February 20-27, 2024, Vancouver,
                  Canada},
  pages        = {12744--12753},
  publisher    = {{AAAI} Press},
  year         = {2024},
  url          = {https://doi.org/10.1609/aaai.v38i11.29170},
  doi          = {10.1609/AAAI.V38I11.29170},
  timestamp    = {Tue, 02 Apr 2024 16:32:09 +0200},
  biburl       = {https://dblp.org/rec/conf/aaai/IzzaIS024.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.